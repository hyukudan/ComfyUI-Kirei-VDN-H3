# ComfyUI Kirei VDN-H3

Native, performance-oriented **VideoDeltaNet (VDN) integration for MiniMax-H3 in
ComfyUI**.

Kirei VDN-H3 patches the MiniMax-H3 model that ComfyUI already loaded. It preserves
ComfyUI's H3 loader, conditioning, audio/video packing, quantized model formats, VAE,
model management and sampler infrastructure while adding the released VDN hybrid
attention branch and a workstation/consumer inference runtime around it.

The project has two non-negotiable goals:

- reproduce the released VDN-H3 architecture and adapter recipe correctly;
- make the added VDN work efficient enough that it does not erase the speed of an
  already-optimized ComfyUI H3 base.

> Model weights are not included. Follow the licenses that apply to MiniMax-H3 and the
> VDN-H3 checkpoints you use.

---

## Quick start

### 1. Install a complete VDN checkpoint

Keep the OpenVDN stage directory intact under `ComfyUI/models/vdn/`.

For the released few-step Stage-DMD model, for example:

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

The stage must contain its specification, linear branch and adapters. Kirei rejects an
incomplete checkpoint rather than silently applying only the pieces it can find.

### 2. Apply VDN to the loaded H3 MODEL

Insert **Kirei Apply VDN-H3** between the MiniMax-H3 model loader and the sampler.

Normal Stage-DMD settings:

```text
profile                 = auto
apply_turbo_adapter     = true
strength                = 1.0
default_adapter_strength = -1   # inherit 1.0
turbo_adapter_strength   = -1   # inherit 1.0
```

The released Stage-DMD generator uses **both** adapters:

```text
default = 1.0
turbo   = 1.0
```

### 3. Use the correct Stage-DMD sampling trajectory

This matters as much as loading the correct checkpoint.

For the released OpenVDN Stage-DMD model use:

```text
sampler       = euler
scheduler     = simple
steps / NFE   = 8
denoise       = 1.0
video shift   = 12.0
audio shift   = 3.0
```

**Do not reuse a base-model `res_multistep` sampler for Stage-DMD.** It changes the
trajectory and can produce hard, patterned or otherwise misleading output even when the
VDN checkpoint itself is applied correctly.

Current ComfyUI MiniMax-H3 handles the video/audio dual flow clocks through its model
sampling implementation, so ordinary stock Euler is appropriate for this path.

---

# Released checkpoint recipes

## Stage-DMD — released 8-NFE model

```text
checkpoint      = stage-dmd-step-250
VDN branch      = ON
default adapter = 1.0
turbo adapter   = 1.0
sampler         = euler
scheduler       = simple
NFE             = 8
denoise         = 1.0
shifts          = video 12 / audio 3
```

The Stage-DMD `turbo` adapter is not merely the external MiniMax-H3 Turbo LoRA dropped
onto VDN. OpenVDN initializes from that few-step adapter and trains the student inside
the VDN model manifold for the released 8-NFE trajectory.

## Stage-B — 50-NFE fidelity/reference model

```text
checkpoint      = stage-b-step-2000
VDN branch      = ON
default adapter = 1.0
turbo adapter   = OFF
sampler         = euler
scheduler       = simple
NFE             = 50
denoise         = 1.0
shifts          = video 12 / audio 3
```

Use Stage-B BF16/reference first when debugging a quality regression. If Stage-B is
already wrong, investigate VDN math, adapter mapping, local-window semantics or AdaLN
handling before blaming Stage-DMD distillation.

---

# Profiles

| Profile | Intended use | Main policy |
| --- | --- | --- |
| `auto` | Recommended runtime | Chooses branch placement from actual VRAM, uses exact calibrated local attention, consumer tiling when needed, resident model optimizations where safe, and follows the loaded H3 precision family for the large VDN projection |
| `max_speed` | Large-VRAM performance experiment | Resident branch, no tiling, merged adapters, compiled pointwise kernels, aggressive projection precision policy, serial tuned execution |
| `balanced` | Manual general-purpose path | Automatic placement/shared compile without forcing the maximum-speed policy |
| `low_vram` | 24 GB-class GPUs | Hybrid branch storage, exact five-frame tiles, factorized LoRA bypass |
| `reference` | Numerical/fidelity oracle | BF16 VDN projection, exact local SDPA, no external attention override, eager/reference-safe execution |
| `compat_reference` | Compatibility debugging | Reference VDN math while allowing ComfyUI's ordinary attention override path |
| `experimental_fp8` | Explicit projection experiment | FP8 VDN projection when the device/runtime probe supports it |
| `workstation_fp8` | Experimental large-VRAM overlap | Resident FP8 branch and optional two-stream branch/softmax overlap |

A profile name is **not** a benchmark result. INT8, FP8 and parallel execution are only
optimizations when they beat the qualified control on the actual GPU/geometry and still
pass the visual quality gate.

---

# What VDN changes

VDN-H3 replaces the video attention behavior with a trained hybrid:

1. **local exact softmax attention** keeps short-range visual detail;
2. a **bidirectional recurrent linear branch** carries context outside that local window.

For each latent frame the linear branch forms compact statistics:

```text
A_t = K_t^T diag(beta_t) K_t
B_t = V_t^T diag(beta_t) K_t
```

The released `vdn_solve` update uses the SPD system around:

```text
I + A_t
```

with Cholesky-based factorization and forward/reverse recurrences. The readout for a
frame uses state representing context outside the same frame's local-softmax coverage,
so the two branches are complementary rather than duplicating the same context.

## Trained invariants preserved

Kirei keeps the checkpoint-declared architecture, including the released settings:

- chunk-aligned local windows;
- `chunk=5`, `radius=1` when declared;
- dense first/last anchors in the trained row/column mode;
- `vdn_solve`;
- alpha bridge;
- K/V short convolution;
- text-state initialization;
- independent softmax and linear gates;
- FP32 A/alpha/Cholesky/recurrent-state islands;
- strict head/block/shape validation.

The released integration shares raw H3 Q/K/V with the VDN branch. Therefore
`linear_head_dim` must match the loaded H3 `head_dim`. Reducing the recurrent state to 64
is a **new-training/checkpoint** project, not a runtime switch.

---

# Exact local attention and quality protection

The trained VDN local windows should remain exact attention.

Normal Kirei VDN paths therefore do **not** route those windows through
`optimized_attention_override`. In particular, a Sage/kitchen quantized attention patch
on the MODEL is not allowed to silently quantize the VDN local window and soften its
output.

Exact does not mean “force the slowest kernel”. Kirei uses
`comfy.ops.scaled_dot_product_attention` when current ComfyUI provides it. Comfy can then
prioritize exact PyTorch kernels such as:

```text
Flash attention -> cuDNN attention -> efficient attention -> math
```

This improves portability/performance, especially on Windows builds, without changing
the exact-attention contract.

`compat_reference` is the explicit mode for experiments where normal Comfy attention
overrides should be allowed.

## Accelerated exact backends

| Backend | Description |
| --- | --- |
| `grouped` | Exact SDPA over contiguous frame groups with identical key sets |
| `flex` | Exact FlexAttention using a cached block mask |
| `flash2` | Exact decomposition: dense global/anchor rows + FA2 varlen local groups |
| `decomposed` | Blackwell-oriented dense + FA4/CuTe varlen decomposition |
| `reference` | Exact grouped oracle, no external override |
| `compat` | Grouped path through Comfy's compatibility dispatcher |
| `auto` | Persistent calibrated exact winner for the current GPU/software/geometry |

For an unseen geometry `auto` can qualify installed exact candidates, persist the winner
and reuse it on later renders. Backend failures are latched so a missing/broken optional
kernel is not retried in every one of 50 blocks.

---

# Mapping the upstream tuned stack onto ComfyUI

OpenVDN's optimized inference accelerates both the new branch and the surrounding H3
block. Kirei reuses the optimizations current ComfyUI already has and implements the
missing pieces natively.

| Component | Runtime provider |
| --- | --- |
| H3 fused QKV | ComfyUI MiniMax-H3 |
| Q/K RMSNorm + RoPE | ComfyUI / comfy-kitchen |
| H3 INT8/ConvRot wide linears | ComfyUI / comfy-kitchen |
| H3 fused SwiGLU input path | ComfyUI |
| VDN window semantics | Kirei |
| grouped/Flex/FA2/FA4 exact dispatch | Kirei |
| temporal 5-tap conv + SiLU/L2Norm | Kirei Triton + fallbacks |
| frame-statistics prologue | Kirei shared runtime |
| scoped TF32 A GEMM | Kirei |
| Cholesky/transition factorization | Kirei |
| preallocated forward/reverse scan | Kirei |
| gather + RMSNorm/output-gate epilogue | Kirei shared compile cache |
| VDN output projection BF16/INT8/FP8 | Kirei |
| H3 pointwise pre/post fusion | Kirei when resident/qualified |
| resident/stream/hybrid branch memory | Kirei + Comfy ModelPatcher |
| low-VRAM factorized LoRA | Kirei |
| resident LoRA merge/requantization | Comfy ModelPatcher |

The rule is to avoid duplicating an optimization Comfy already performs well, while
making the new VDN work participate correctly in Comfy's precision and memory lifecycle.

---

# VDN projection precision

The dominant VDN branch weight is `to_out_linear`.

```text
projection_precision = auto | bf16 | int8 | fp8
```

## BF16

Use BF16 first when qualifying architecture and output quality. It removes projection
quantization as a confounding variable.

## INT8 / ConvRot

On compatible current ComfyUI builds, Kirei can convert the VDN projection with the same
`TensorWiseINT8Layout` / ConvRot family used by the H3 base and execute it with
comfy-kitchen.

Sensitive VDN state is not converted to INT8.

INT8 is a Comfy-specific experiment relative to the OpenVDN release. If BF16 is faster
on your GPU/geometry, INT8 is simply not an optimization for that case.

## FP8

FP8 is separately available for the large VDN projection. Capability is probed on the
actual device/runtime before the model is installed. A failed capability path falls back
to BF16 rather than dying halfway through a render.

Quantized paths can alter the denoising trajectory. Qualify final output, not only local
GEMM error.

---

# LoRA and adapter handling

The Stage-B and Stage-DMD adapters are converted from their upstream target names to the
native Comfy H3 module layout, including:

- fused Q/K/V slices;
- text-refiner targets;
- SwiGLU half ordering;
- native `mlp.fc2` exception;
- pruned/curve AdaLN terms.

## Low-VRAM factorized bypass

Kirei keeps A/B factors low-rank. Terms targeting the same native module share one down
projection and terms targeting the same output slice share one up projection. Default +
turbo QKV can therefore run approximately as:

```text
1 shared down
+ 1 Q up
+ 1 K up
+ 1 V up
```

without constructing dense `B @ A` weights.

## Resident merge/requantization

For resident quantized workstation paths, eligible adapters can be merged through
ComfyUI's ModelPatcher during model loading so the hot forward retains native quantized
linears rather than paying extra low-rank GEMMs on every block/step.

## Independent strengths

Advanced controls:

```text
default_adapter_strength
turbo_adapter_strength
```

The canonical Stage-DMD release benchmark requires both at **1.0**.

The Runtime Report records:

```text
adapters.active
adapters.strengths
adapters.lora_mode
```

so benchmark results can reject a missing adapter or wrong strength.

---

# Pruned / curve AdaLN bases

Some pruned H3 checkpoints replace full AdaLN time projections with a compact shared
curve basis. Kirei does not silently drop the original adapter's AdaLN terms.

Curve mode is detected using either:

- `use_adaln_curves`, or
- the structurally collapsed input width of `blocks[0].adaln_proj.linear`.

This second check matters for converted checkpoints whose flag is not reliable.

The original full-space LoRA delta is re-injected at runtime using
`h3_silu_temb_grid.safetensors` and the model's `adaln_t_table`.

The e-grid is searched in the checkpoint/model roots first and the legacy
`ComfyUI-MiniMax-H3-Turbo` sibling location last. If the checkpoint structurally needs
curve reinjection but the required data is missing, Kirei fails clearly instead of
quietly removing those adapter terms.

---

# Linear branch memory and execution

## Serial tuned path

In inference the VDN branch consumes raw Q/K/V before native in-place Q/K norm + RoPE:

```text
native qkv_proj
   ├─ raw Q/K/V -> VDN branch -> projected linear delta
   └─ Q/K norm + RoPE -> local exact softmax -> H3 out_proj
                                           + linear delta
```

This avoids holding three full raw-video copies while softmax runs.

## Tiled exact path

On constrained GPUs, K/V are processed in frame tiles with the exact two-frame halo
needed by the 5-tap temporal convolution. Only compact A/B/alpha frame statistics survive
the first pass; Q/readout/output projection are then processed tile-by-tile after the
global recurrence.

This reduces activation VRAM without changing the trained recurrence.

## Scan

The tuned recurrence uses:

- batched transition factorization;
- preallocated prefix/suffix state banks;
- one `torch.baddbmm(..., out=...)` per frame/direction;
- no `.item()` synchronization in the hot path.

The dependent scan is deliberately **not** compiled into one giant Inductor graph.

## Optional parallel execution

`branch_execution=parallel` can overlap the independent VDN branch and local softmax on
a large-VRAM GPU. It also increases activation lifetime and resource contention, so it
remains an explicit experiment rather than an assumed default.

---

# Branch weight placement

## `resident`

All VDN branch tensors participate in Comfy's model lifecycle on GPU. Preferred on a
96 GB-class workstation when activation headroom remains sufficient.

## `hybrid`

Small branch tensors stay resident while the dominant output projection is streamed.
Normally the useful compromise for 24 GB-class cards.

## `stream`

Branch tensors remain CPU masters and are copied through reusable CUDA staging slots.

Streaming includes:

- independent buffer allocation vs content validity;
- per-slot block identity + `valid_keys`;
- two reusable staging buffers;
- reusable ready/consumed CUDA events;
- dedicated prefetch stream;
- N+1 prefetch;
- cyclic block 49 -> block 0 prefetch;
- Comfy pinned-memory policy by default;
- H2D and wait telemetry.

This avoids the stale-buffer bug where an allocated tensor name could otherwise be
mistaken for data valid for the current block after a partial load.

---

# Shared compile/runtime caches

One VDN state owns caches shared by all 50 H3 blocks:

- gather indices;
- compiled static operations;
- temporal kernels;
- attention masks/plans/calibration;
- failure latches.

Controls:

```text
kernel_backend = auto | triton | conv1d | eager
compile_policy = auto | off | shared | reduce_overhead | max_autotune
```

Compile failure falls back and is remembered. The runtime does not repeatedly compile
the same known-broken shape.

---

# GPU policy

## RTX PRO 6000 Blackwell 96 GB

Typical candidates:

- resident branch;
- untiled activations;
- exact calibrated attention, including FA4 when available;
- merged adapters on qualified quantized resident paths;
- pointwise block fusion;
- BF16 vs INT8 vs FP8 benchmarked explicitly;
- serial path as control;
- parallel path only as a measured experiment.

## RTX 4090 24 GB

Typical candidates:

- hybrid placement;
- five-frame exact tiles;
- factorized LoRA bypass;
- grouped/Flex and FA2 where installed;
- stream as lower-VRAM fallback;
- quantized projection only when it wins the actual-device benchmark.

`auto` is a runtime policy, not a promise that every selected micro-path wins every
geometry. Use the benchmark suite to qualify it on your machine.

---

# Benchmarking — do not reuse production sampler widgets

Kirei includes a recipe-aware benchmark path specifically to prevent configuration drift.

Use:

```text
Kirei Benchmark Scenario
        │ scenario_id
        ▼
Kirei Benchmark Sampling ◄──── MODEL
        │
        ├── SAMPLER ─────────────────┐
        ├── SIGMAS ──────────────────┤
        └── recipe_token             │
                 │                   │
                 ▼                   │
Kirei Benchmark Start ◄────── MODEL │
        │ MODEL                      │
        └────────────────────────────┼──► SamplerCustomAdvanced
                                    │
SamplerCustomAdvanced ◄─────────────┘
        │ LATENT
        ▼
Kirei Benchmark End
```

**Kirei Benchmark Sampling owns the real sampler and sigma schedule.** For Stage-DMD it
constructs Euler + simple + 8 NFE itself. `Benchmark Start` refuses a hand-made/unverified
recipe token.

The result recorder also validates:

- sampler;
- scheduler;
- NFE;
- denoise;
- model video/audio shifts;
- checkpoint recipe;
- profile;
- actual projection precision;
- active adapters and strengths.

A run cannot therefore be labelled `VDN BF16 canonical` while actually using
`res_multistep`, beta scheduling, six steps, wrong adapters or a silent INT8 fallback.

See [`benchmarks/README.md`](benchmarks/README.md) and
[`docs/VALIDATION.md`](docs/VALIDATION.md) for the full protocol.

## Active benchmark tiers

| Tier | Geometry | Purpose |
| --- | --- | --- |
| Quick | 608×352 / 121 requested frames | frequent regression |
| Detail | 960×544 / 121 | micro-detail/quality A/B |
| Primary | 608×352 / 241 | repeated long-video performance decision |
| Stress | 608×352 / 401 | long-sequence crossover/scaling |
| Canonical quality | 1344×768 / 345 | Stage-B fidelity + Stage-DMD quality gate |

For the 8-NFE quality groups the clean control is Larry v4 at strength 1.0 with the same
Euler/simple/8 trajectory as VDN. The locally preferred hybrid/Sage/Larry-0.6/Euler-beta
six-step workflow is retained separately as a visual reference, not mixed into canonical
speed claims.

---

# Runtime Report

Connect **Kirei VDN-H3 Runtime Report** downstream of sampling when profiling.

It reports, among other fields:

```text
checkpoint / checkpoint_recipe
adapters.active / adapters.strengths / adapters.lora_mode
profile
base_precision
projection.precision
branch_mode / branch_execution
branch_storage
last_layout
attention_requested / attention_last
attention_calibration
block_fusion / block_fusion_error
kernel_backend / compile_policy / tile_frames
lora_factors / curve_factors
diagnostics
CUDA memory
performance_analysis
```

Use the report to verify what actually executed before interpreting a timing number.

---

# CUDA probes

General optimized-runtime probe:

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-core-gpu0.json
```

Workstation/consumer precision and memory probe:

```bash
python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic-gpu0.json
```

Run them separately on each target GPU. Synthetic probes qualify kernels; they do not
replace the real Comfy generation benchmark.

---

# First run vs steady state

The first VDN render can include:

- checkpoint parsing;
- auxiliary model registration;
- LoRA merge/requantization;
- INT8/FP8 projection conversion;
- Triton compile;
- Inductor compile;
- attention backend calibration;
- allocator setup.

Do not compare that cold run against a warmed control.

The standard protocol records cold separately, performs at least one warm-up and uses the
median of five warm sampler-only runs by default. VAE decode/MP4 encoding are separate
end-to-end metrics.

---

# When quality looks wrong

Check in this order:

1. **sampling plan** — Stage-DMD must say Euler/simple/8/denoise1;
2. **adapters** — `default=1.0 + turbo=1.0`;
3. **BF16 control** — remove INT8/FP8 as a confounder;
4. **local attention** — no Sage/kitchen quantized override on VDN windows;
5. **curve AdaLN** — required e-grid terms must be active, not skipped;
6. **Stage-B/50 reference** — determines whether the issue predates Stage-DMD Turbo.

The requested output frame count can be internally aligned by H3 to its supported
temporal grid. That alone is not a quality failure.

---

# When VDN looks slower

Treat it as a profiler result, not as an acceptable property of an “optimization”.

Check:

- cold vs warm;
- actual attention backend/calibration hit;
- `linear_branch_total_ms` vs `softmax_total_ms`;
- projection precision and whether that precision actually wins;
- branch H2D waits;
- block-fusion fallback;
- branch placement;
- tiling/parallel policy;
- identical sampler/scheduler/NFE/denoise between technical variants.

If INT8 is slower than BF16, keep BF16 for that GPU/geometry until the INT8 path is
improved. If VDN still loses materially in the 401-frame stress workload, investigate the
hot path rather than explaining it away as a short-video crossover.

---

# Advanced inputs

| Input | Meaning |
| --- | --- |
| `branch_mode` | `auto`, `resident`, `hybrid`, `stream` |
| `branch_execution` | serial control or explicit experimental parallel |
| `lora_mode` | automatic, factorized bypass, or merge |
| `attention_backend` | auto/grouped/flex/flash2/decomposed/reference/compat |
| `kernel_backend` | auto/triton/conv1d/eager |
| `compile_policy` | auto/off/shared/reduce_overhead/max_autotune |
| `tile_frames` | 0 lets the profile choose; positive forces exact frame tiling |
| `pin_strategy` | Comfy-aware pinning, none, or aggressive override |
| `projection_precision` | auto/bf16/int8/fp8 |
| `default_adapter_strength` | independent Stage-B adapter strength |
| `turbo_adapter_strength` | independent Stage-DMD adapter strength |
| `strict_validation` | reject incompatible checkpoint/base early |
| `diagnostics` | collect CUDA-event stage telemetry |

Use the named `reference` profile rather than trying to reconstruct an oracle manually
from advanced switches.

---

# Compatibility and lifecycle

- VDN branch weights are registered as Comfy additional models.
- LoRA bypass and curve factors are also visible to model lifecycle/accounting.
- Applying VDN twice is rejected.
- patch collisions are rejected rather than silently overwritten.
- VDN runtime state is single-selected-GPU; distributed/Ulysses needs explicit sharding
  logic and is not emulated by shallow ModelPatcher cloning.
- **Kirei Release VDN-H3 Weights** releases VDN weights/caches/auxiliaries.
- the legacy node id remains for older saved workflows.

---

# Upstream reference

The VDN mathematics, released checkpoint architecture and tuned inference design come
from **OpenVDN / VDN-Minimax-H3**:

- `https://github.com/OpenVDN/vdn-minimax-h3`
- `https://openvdn.github.io/`

Kirei's Comfy-specific work focuses on native model patching, memory management,
quantized H3 compatibility, exact attention dispatch, profiling and workstation/consumer
execution policies.

See `THIRD_PARTY.md` and `NOTICE` for attribution details.

---

# Performance claims

Do not infer a speedup from a profile name or a theoretical complexity argument.
A reproducible claim should record at least:

```text
GPU + VRAM
CUDA / PyTorch / ComfyUI revision
H3 base checkpoint and precision/layout
VDN checkpoint
active adapters + strengths
sampler + scheduler + NFE + denoise
video/audio shifts
resolution + requested frames
actual latent/packed geometry
VDN projection precision
attention backend + calibration hit
branch placement + execution mode
block fusion status
cold time
warm median sampler time
peak VRAM
quality status
```

The benchmark suite exists to make those conditions machine-checkable rather than
relying on workflow titles or remembered widget values.

---

## License

Repository code follows the license declared in this repository. Adapted upstream
algorithms retain their required notices and attribution. Model weights are distributed
separately and may be subject to different terms.
