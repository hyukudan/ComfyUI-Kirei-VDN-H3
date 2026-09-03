# VDN-H3 runtime architecture

This document describes the current Kirei VDN-H3 runtime as an implementation contract:
mathematical invariants, tensor lifetime, memory placement, adapter handling, exact
attention and backend policy.

---

## 1. Integration boundary

Kirei patches ComfyUI's native MiniMax-H3 model rather than loading a parallel Diffusers
transformer.

Reused from current ComfyUI:

- H3 checkpoint and quantized loaders;
- packed text/audio/video model layout;
- native fused QKV projection;
- Q/K RMSNorm + RoPE kernels;
- native output projection and MLP paths;
- ModelPatcher load/offload/requantization lifecycle;
- model sampling / audio-video shift handling;
- VAE/audio/conditioning pipeline.

Added by Kirei:

- released VDN local-window semantics;
- bidirectional recurrent linear branch;
- softmax/linear gates;
- VDN branch storage and placement policies;
- upstream-to-native adapter conversion;
- factorized adapter runtime / resident merge policy;
- pruned AdaLN curve reinjection;
- exact attention backends and calibration;
- temporal/frame-statistics kernels;
- block pointwise fusion;
- profiling and recipe-aware benchmark infrastructure.

No ComfyUI core file is modified.

The benchmark infrastructure owns sampler/sigmas when measuring canonical release
recipes, but normal model application deliberately remains independent from the user's
sampler graph.

---

## 2. Mathematical contract

For one latent video frame after the trained K/V feature transform:

```text
A_t = K_t^T diag(beta_t) K_t
B_t = V_t^T diag(beta_t) K_t
```

The released `vdn_solve` transition is built around the SPD system `I + A_t`. Kirei
retains the Cholesky-based formulation and runs one recurrence forward and one backward.

The linear readout for frame `t` gathers recurrent state corresponding to context outside
that frame's local-softmax coverage.

Precision-sensitive values remain FP32 even when H3 compute is BF16:

- `alpha.A_log`;
- `alpha.dt_bias`;
- A statistics;
- Cholesky/inverse construction;
- recurrent state banks.

### Released structural invariants

Checkpoint-declared values are authoritative. The released lineage includes:

- `chunk=5`;
- `radius=1`;
- anchor mode `both`;
- `vdn_solve`;
- alpha bridge;
- K/V short convolution;
- text-state initialization;
- softmax and linear gates;
- `a_fp32=true`.

The current native port shares raw H3 Q/K/V with the VDN branch. Therefore released
`linear_head_dim` must equal H3 attention `head_dim` (128 on the published geometry).
Different state size requires a separately trained checkpoint.

---

## 3. Packed layout

ComfyUI's packed H3 request is translated into an immutable `VDNLayout` and published
for one diffusion-model forward through a `ContextVar`.

It records:

- total packed sequence length;
- video start/end rows;
- latent frame count;
- tokens per frame;
- spatial frame grid;
- text segment;
- exact per-frame window bounds;
- anchor mode;
- full-cover status.

Request-local publication avoids leaking layout state between nested/concurrent forwards.

---

## 4. Attention dataflow

### Tuned serial inference

```text
x
│
├─ native fused qkv_proj
│      │
│      ├─ raw Q/K/V views ─────── VDN branch ───── projected linear delta
│      │
│      └─ Q/K norm + RoPE ─────── local exact softmax ───── H3 out_proj
│                                                        │
└────────────────────────────────────────────────────────┴── add VDN delta
```

The linear branch runs before native in-place Q/K normalization/RoPE. Once the branch is
projected into H3 hidden size, raw video Q/K/V no longer need to remain live through
softmax.

### Reference/autograd path

The reference path remains intentionally conservative:

- compact raw video/text copies are allowed;
- exact grouped SDPA is used;
- reference recurrence is autograd-safe;
- quantized VDN projection is disabled;
- inference-only block fusion/shortcuts are disabled.

---

## 5. Linear branch execution

### Untiled

1. prepare Q/K/V features;
2. compute beta, alpha, A and B;
3. factor transitions in batch;
4. fill preallocated prefix/suffix banks;
5. gather complementary state;
6. read out and apply RMSNorm/output gate;
7. project only rows that can have non-zero VDN contribution.

### Tiled

Tiling changes activation materialization, not the recurrence.

#### Statistics pass

For each K/V tile plus the required two-frame temporal halo:

```text
raw K/V
 -> optional depthwise spatial conv
 -> exact 5-tap temporal conv
 -> SiLU / L2Norm
 -> beta
 -> A/B
```

Only center-frame statistics are retained.

#### Global scan

After all compact frame statistics exist, the complete bidirectional recurrence runs
across the sequence.

#### Readout pass

For each Q/x tile:

```text
Q feature
 -> complementary-state gather
 -> readout
 -> RMSNorm/output gate
 -> BF16/INT8/FP8 to_out_linear
 -> output slice
```

This removes full prepared Q/K/V, gate and readout tensors from the long-lived working
set.

---

## 6. Recurrence design

The tuned scan keeps the same strategy as the optimized OpenVDN inference path:

- transition/injection factorization outside the dependent recurrence;
- preallocated prefix/suffix state banks;
- one `torch.baddbmm(..., out=...)` per frame and direction;
- no GPU `.item()` synchronization in the loop.

The dependent scan is deliberately **not** compiled as one huge Inductor graph. At the
released 128-dimensional state, the recurrence is launch-bound and serial; a giant
compiled graph increases startup/specialization complexity without an established
steady-state win.

Associative/prefix-scan reformulations remain research experiments because they trade
serial launches for additional matrix products and temporary storage.

---

## 7. Shared runtime caches

One `SharedBranchRuntime` belongs to one patched `VDNState` and is reused by all 50 H3
blocks.

It owns/reuses:

- gather indices;
- compiled static operations;
- temporal-kernel dispatch state;
- failure latches.

The attention cache separately owns:

- grouped/Flex mask data;
- decomposed varlen plans;
- Flex compiled call;
- persistent calibration lookup;
- backend failure latches.

This prevents equivalent compiler wrappers/plans from being recreated 50 times.

---

## 8. Temporal and frame-statistics kernels

Spatial short convolution stays on depthwise Conv2d so cuDNN can select its native
kernel.

Temporal short convolution dispatches through an exact fused Triton implementation of:

```text
5-tap temporal depthwise conv -> SiLU -> optional L2Norm
```

with exact Conv1d/eager fallbacks.

Optimized feature output can be written directly as:

```text
[F, H, S, D]
```

which is the layout consumed by frame-statistics GEMMs.

The shared frame-statistics prologue:

- creates contiguous K in the needed layout;
- forms K/A in FP32;
- applies beta weighting;
- scopes TF32 only around the A GEMM during inference;
- symmetrizes A;
- computes B and promotes recurrent values to FP32.

Cholesky/recurrence are outside autocast.

---

## 9. Branch weight memory

VDN branch weights are represented as real non-trainable module parameters and attached
through Comfy additional ModelPatchers.

### Resident

All branch weights participate in Comfy load/offload accounting on GPU.

### Stream

CPU masters feed two reusable CUDA staging slots.

### Hybrid

Small branch tensors remain resident while the dominant output projection representation
is streamed.

### Slot correctness

A slot tracks independently:

- allocated buffers;
- current block identity;
- `valid_keys` for that identity;
- ready event;
- consumed event.

Changing block identity invalidates contents even when same-shaped buffers remain
allocated. Allocation is never treated as proof of valid data.

### Prefetch

- N+1 can be scheduled while N computes;
- the last block can prefetch block 0 for the next NFE;
- pinned memory respects Comfy's policy unless explicitly overridden.

---

## 10. Adapter runtime

### Target conversion

Upstream adapter names are mapped to native H3, including:

- block prefix conversion;
- fused Q/K/V slices;
- text refiner;
- SwiGLU half swap;
- native `mlp.fc2` exception;
- final/pruned AdaLN targets.

PEFT `alpha/rank` scaling and mixed ranks are preserved.

### Composite low-rank bypass

All A/down terms targeting one native module are concatenated into one down projection.
Terms sharing an output slice concatenate scaled B/up matrices column-wise:

```text
[B1*s1 | B2*s2] @ [A1(x); A2(x)]
 = B1 A1(x) s1 + B2 A2(x) s2
```

Default+turbo fused QKV therefore needs one shared down GEMM and one up GEMM per Q/K/V
slice instead of independent pairs for every adapter term.

### `mlp.fc2` exact bypass

Native H3 consumes `mlp.fc2` through `comfy.ops.linear_input_act`, which folds the SwiGLU
into the INT8 kernel and never calls `fc2.forward`. Those factors are hooked on the
parent MLP instead: the native fc2 GEMM (fused/quantized or plain) is kept and the exact
low-rank term is added from the same SwiGLU activation. No target is ever forced into a
weight merge. Merging into a quantized weight requantizes it and rounds part of the
update away, which is exactly the softening `lora_mode=bypass` exists to avoid.

### Adapter policy

`auto` merges on BF16 bases (one fp32 delta, one rounding, the fold OpenVDN performs
before inference, no low-rank GEMMs left in the hot path) and keeps the activation-space
bypass on every quantized base (INT8, FP8, NVFP4, MXFP8, W4). `max_speed` merges
everywhere and must pass its own quality gate.

### Runtime recipe metadata

After successful application `VDNState` records:

```text
adapters.active
adapters.strengths
adapters.lora_mode
adapters.reports
```

The benchmark recorder uses this metadata to enforce Stage-DMD `default=1 + turbo=1` and
Stage-B `default=1`.

---

## 11. Pruned / curve AdaLN

Curve-mode H3 may collapse `adaln_proj.linear` onto a small shared coordinate basis.
Detection uses either:

1. `use_adaln_curves`, or
2. a structurally small input width on the first block's AdaLN projection.

The structural check handles converted checkpoints whose flag is missing/unreliable.

Curve LoRA terms are **not skipped** in merge-oriented profiles. They remain a distinct
runtime curve category and are re-injected using:

- the model's `adaln_t_table`;
- `h3_silu_temb_grid.safetensors`;
- the original low-rank factors.

The e-grid search prefers checkpoint-local/model-root locations and retains the legacy
MiniMax-H3-Turbo sibling path only as a fallback.

---

## 11b. AdaLN fp32 activation

ComfyUI casts the fp32 time embedding to the compute dtype before every block's AdaLN
SiLU. OpenVDN patched diffusers to activate in fp32 and cast afterwards, after measuring
a 3.5e-3 norm-relative error on the modulation that biases every block identically at
every step; the released checkpoints were trained under that patch.

Kirei object-patches `time_embedder.forward` to keep the fp32 embedding for the current
forward and replaces every `adaln_proj.forward` (50 blocks plus the final layer) with a
body that activates that copy and projects in the model dtype. Curve-form bases (no
SiLU, fp32 AdaLN) are skipped. Switch: `adaln_fp32`; state: Runtime Report `adaln_fp32`.

---

## 12. Exact local attention

VDN local windows are exact attention in normal/reference paths.

### Grouped exact SDPA

Frames with identical allowed key sets become contiguous query runs. Each run performs
one exact SDPA against its selected global/video keys.

When available, Kirei calls `comfy.ops.scaled_dot_product_attention`. This preserves
exact numerics while allowing current ComfyUI to prioritize the platform's exact
PyTorch backend:

```text
Flash -> cuDNN -> efficient -> math
```

The grouped/reference path does **not** pass `transformer_options` into Comfy's
`optimized_attention`, so Sage/kitchen quantized overrides cannot silently modify the
trained VDN local window.

`compat` is the explicit exception and exists for compatibility experiments.

### Flex

The exact pattern is represented by a cached block mask and a model-owned compiled Flex
call.

### Flex compilation

`flex_attention` is compiled once per cache with `dynamic=True`, and the window mask
closure captures its geometry as tensors rather than Python ints, so neither the kernel
nor `create_block_mask` recompiles per packed length. Dynamo's recompile limit is raised
to at least 64 before the first compile: past that limit dynamo runs the function
eagerly, and eager Flex materialises the full S x S score matrix.
`tests/probe_flex_cuda.py` pushes twelve lengths through one cache with
`fail_on_recompile_limit_hit` set to prove it.

### FA2 / FA4 decomposition

A cached plan separates:

- dense global/anchor rows;
- local window groups.

Local groups run through varlen FA2 (`flash2`) or CuTe/FA4 (`decomposed`) and scatter back
to packed row order. Dense rows remain exact SDPA.

### Auto calibration

`auto` checks a persistent calibration signature before heuristics. The signature
contains the GPU name and capability, the software identity that decides which backends
exist and how fast they run (node version, torch, CUDA, cuDNN, driver, Triton,
flash-attn 2 and flash-attn-4 versions, the installed backend inventory) and the exact
packed geometry. Any change re-measures instead of trusting an old winner; store files
of an older schema are ignored. Only exact candidates that pass parity against the
grouped oracle can win. The resolution reason (`explicit`, `calibrated`, `flex` guard,
`autotuned`, `heuristic`) is rewritten on every call and reported.

### GPU families

`window.gpu_family` classifies the device once and `window.fa4_kernel` names the
flash-attn-4 generation it would run: `tcgen05` on datacenter Blackwell (sm_100/sm_103),
`wgmma` on Hopper (sm_90), `mma_sync` on consumer and workstation Blackwell (sm_120:
RTX 50xx, RTX PRO 6000), Ada and Ampere. Only the first two skip the calibration queue;
the mma.sync generation is an SM80-class kernel with 99 KB of shared memory on sm_120, so
it competes with grouped SDPA, Flex and FA2 and wins only by measurement. Before
autotuning, `auto` also estimates the K/V copy volume
grouped attention would need for the layout (every global row is copied into every
window group) and selects Flex outright when the copies would exceed the guard; the
reason is reported as `attention_calibration.dispatch_reason`.

---

## 13. Projection precision

The dominant VDN output projection can be:

```text
bf16 | int8 | fp8
```

### INT8 / ConvRot

Uses current Comfy quantized-tensor / comfy-kitchen execution where supported. Recurrent
state and narrow/sensitive VDN operations stay BF16/FP32.

### FP8

Stores an actual quantized projection representation plus scale; it does not retain a
hidden BF16 duplicate for quantized interior blocks. Actual-device scaled-matmul support
is probed before installation.

`detect_base_precision` recognises every comfy-kitchen storage family (INT8, FP8,
NVFP4, MXFP8, W4A4, W4A8). Families without a matching VDN projection keep the BF16
projection; the base being quantized still selects the exact adapter bypass.

Precision policy may follow an explicitly selected scenario/profile or an already
quantized base. Benchmark labels are validated against the **resolved** Runtime Report,
so an INT8/FP8 fallback cannot be recorded under the wrong name.

---

## 14. H3 pointwise fusion

For qualified resident inference, Kirei can compile shared block bodies for:

```text
RMSNorm + AdaLN modulation
residual + gate * branch output
```

The same compiled shapes are reused across all 50 blocks.

If a compiled body fails, block fusion is disabled/falls back to the native Comfy block
from the original residual. There is no intentionally slower eager imitation of the
fused path.

---

## 15. Parallel branch experiment

The local softmax and recurrent branch are independent until their output sum. A
large-VRAM experimental path can therefore copy compact raw Q/K/V and run the VDN branch
on another CUDA stream.

The path is exact but not assumed faster: concurrent tensor-core/attention workloads can
compete for the same SM resources. Serial remains the control until target-device
benchmarks prove otherwise.

---

## 16. Diagnostics and benchmark contract

CUDA stage diagnostics use recorded events and resolve them at report time instead of
synchronizing around every scope.

The Runtime Report exposes:

- checkpoint recipe;
- adapter recipe and `adaln_fp32`;
- GPU family and the attention dispatch reason;
- packed-layout row breakdown (video / text / other global rows);
- profile/base/projection precision;
- branch placement/execution;
- attention/backend calibration/failures;
- block fusion;
- actual packed geometry;
- branch streaming telemetry;
- LoRA/curve storage;
- stage timings and CUDA memory.

Canonical benchmark sampling is generated from `benchmarks/scenarios.json` by
`Kirei Benchmark Sampling`. Stage-DMD is therefore mechanically tied to:

```text
Euler / simple / 8 NFE / denoise 1.0 / shifts 12,3
```

and Stage-B to the same trajectory at 50 NFE. `Benchmark Start` requires the verified
recipe token so a saved `res_multistep`/beta/six-step widget cannot silently contaminate
a canonical result.

---

## 17. Single-GPU lifecycle

Branch, LoRA and curve auxiliaries use Comfy load/offload accounting.

A patched VDN runtime represents one selected compute GPU. MultiGPU shallow cloning is
rejected because copied Python closures/caches would not constitute a valid distributed
VDN execution. A real Ulysses/distributed port requires explicit sharding and
communication logic.
