# ComfyUI Kirei VDN-H3

Optimized native ComfyUI integration of **VideoDeltaNet (VDN) for MiniMax-H3**.

The node patches ComfyUI's existing MiniMax-H3 model. It does **not** replace the H3
loader, sampler, VAE, audio path, conditioning stack or quantized base-model support.
The released VDN branch and adapters are translated into ComfyUI-native runtime
patches and managed as additional models.

The default goal is simple:

- keep the released VDN-H3 mathematics intact;
- make the 96 GB workstation path fast and uncomplicated;
- make 24 GB consumer GPUs practical by reducing live activations and PCIe traffic;
- never enable a precision-changing optimization silently.

## Main node

Use **Kirei Apply VDN-H3** between the MiniMax-H3 model loader and the sampler.

Required inputs:

- `model`
- `vdn_checkpoint`
- `profile`
- `apply_turbo_adapter`
- `strength`

For most workflows, leave the advanced inputs at their defaults and use `profile=auto`.

## Profiles

| Profile | Intended use | Runtime policy |
| --- | --- | --- |
| `auto` | Recommended | Chooses resident / hybrid / streamed branch placement from actual free VRAM, tiles the linear branch on 24 GB-class GPUs, uses exact factorized LoRA and the best safe attention backend |
| `max_speed` | Large-VRAM workstation | Forces resident branch weights, no branch tiling, shared compiled kernels with `reduce-overhead`; use when the complete H3 + VDN working set comfortably fits |
| `balanced` | General manual profile | Optimized inference with automatic branch placement and shared compilation |
| `low_vram` | 24 GB-class GPUs | Hybrid branch placement plus exact 5-frame tiled linear execution |
| `reference` | Numerical oracle | Exact SDPA with no Comfy attention override, BF16/FP32 VDN math, factorized LoRA bypass, eager/autograd-safe branch |
| `compat_reference` | Comfy compatibility comparison | Reference branch math while allowing ComfyUI's attention dispatch/override path |
| `experimental_fp8` | Explicit experiment | Quantizes only the dominant VDN output projection when the active GPU/PyTorch `_scaled_mm` path passes a runtime probe; never selected by `auto` |

### Expected policy on common cards

**RTX PRO 6000 Blackwell 96 GB**

`auto` normally selects resident branch weights when there is enough current free VRAM.
This avoids H2D traffic during the 50 transformed DiT blocks and leaves the tiled path
disabled. If `flash_attn.cute`/FA4 is present, Blackwell can also use the decomposed
varlen attention path; otherwise `auto` falls back safely.

**RTX 4090 24 GB**

`auto` normally selects `hybrid` branch storage and 5-frame tiles. The small VDN branch
weights remain resident while the very large `to_out_linear` projection is double-
buffered from CPU. FA2 is available as an explicit or calibrated backend but is not
blindly selected, because gather overhead can erase its kernel advantage on Ada.

These are policies, not fixed benchmark claims. Use the runtime report and CUDA probes
below on the actual machine and geometry.

---

# What VDN-H3 is doing

VDN-H3 combines two complementary attention paths inside each transformed H3 DiT
block:

1. a **local softmax branch**, which preserves high-detail short-range interactions;
2. a **bidirectional linear recurrent branch**, which carries information outside the
   local softmax window.

For each video frame, the linear branch condenses its K/V tokens into frame statistics:

```text
A_t = K_t^T diag(beta_t) K_t
B_t = V_t^T diag(beta_t) K_t
```

The recurrent state then applies the VDN solve around `(I + A_t)^-1`, with a forward
and reverse scan. The released model uses a 128-dimensional per-head state, FP32 state
statistics/recurrence and a Cholesky-based solve.

The local softmax window and the linear state are deliberately complementary. The
linear readout gathers state from **outside** the frame's local softmax coverage instead
of counting the same context twice.

## Released-model invariants preserved by this integration

The runtime keeps the checkpoint's trained semantics unchanged:

- chunk-aligned local softmax window;
- released radius/chunk configuration (`chunk=5`, `radius=1` for the published stage);
- dense first/last frame anchors in the trained row/column mode;
- `vdn_solve` recurrence;
- multiplicative alpha bridge;
- prompt/text initialization of both recurrent directions when enabled by the spec;
- FP32 `A`, alpha parameters, Cholesky solve and recurrent state;
- softmax and linear output gates;
- K/V short convolution when declared by the checkpoint;
- strict checkpoint inventory and shape validation.

The current released integration shares H3's raw native Q/K/V with the linear branch.
For that reason `linear_head_dim` must equal the loaded H3 attention `head_dim`. A
smaller state dimension such as 64 is a **different trained model**, not a runtime
switch.

---

# Memory architecture

## 1. Linear branch before softmax

The optimized inference path computes the VDN linear branch **before** RoPE/local
softmax while the native fused QKV buffer already exists.

Conceptually:

```text
native fused QKV
    │
    ├── raw Q/K/V views ──> VDN linear branch ──> projected linear delta
    │
    └── native Q/K norm + RoPE ──> local softmax ──> H3 out_proj
                                              │
                                              └── + linear delta
```

This is mathematically equivalent because the two branches are independent until the
final sum, but it changes tensor lifetime substantially.

The older spelling needed compact full-video Q/K/V clones to survive while the fused
Q/K buffer was mutated by native RMSNorm+RoPE. The optimized inference path consumes
raw Q/K/V first and retains only the already-projected linear output while softmax runs.
For long videos this can remove several GiB of simultaneous raw-feature storage.

`reference` keeps the conservative deferred/autograd-safe path with compact copies, so
it remains useful as an oracle.

## 2. Exact tiled branch for consumer GPUs

`tile_frames > 0` splits the linear branch without changing the recurrence math.

First pass, by frame tile:

```text
raw K/V (+ temporal halo)
    -> spatial/temporal short conv
    -> SiLU / normalization
    -> beta
    -> A_t / B_t frame statistics
```

Only the compact per-frame statistics survive.

Then the exact forward/reverse scans run once.

Second pass, by frame tile:

```text
raw Q (+ halo when needed)
    -> feature map
    -> gather outside-window state
    -> output gate + RMSNorm
    -> to_out_linear
    -> write directly into the final linear delta
```

The first/last dense anchors are removed before the dominant projection rather than
being projected as guaranteed zeros.

`low_vram` uses 5-frame tiles by default. `auto` also uses 5-frame tiles on <=32 GB
CUDA devices.

## 3. Branch weight placement

Three modes are available:

### `resident`

All VDN branch weights are Comfy-managed GPU auxiliary weights.

Best when memory is abundant.

### `stream`

The complete branch remains on CPU and requested block weights are staged through two
reusable GPU slots.

### `hybrid`

Designed for consumer cards. The small branch tensors remain resident while only the
large projection is streamed.

For the released H3 geometry, `to_out_linear` dominates the branch storage, so hybrid
mode removes most H2D traffic without spending several additional GiB of persistent
VRAM.

Streaming uses:

- explicit per-block valid-key tracking;
- two reusable slots;
- reusable CUDA `ready` / `consumed` events;
- a dedicated prefetch stream;
- early prefetch before native QKV;
- prefetch of block N+1 while block N computes;
- cyclic block 49 -> block 0 prefetch for the next denoising step;
- ComfyUI's pinned-memory policy by default.

`pin_strategy=all` is an explicit override for machines with abundant host RAM. `auto`
and `comfy` do not bypass ComfyUI's pinned-memory budget.

---

# Linear kernels

Kernel choice and compilation are separate controls.

## `kernel_backend`

- `auto`
- `triton`
- `conv1d`
- `eager`

The temporal 5-tap path prefers a fused Triton implementation when available, with
Conv1d and eager fallbacks. K/V/Q can be written directly in frame/head/token/head-dim
layout so frame statistics do not need an additional full-feature repack.

The spatial depthwise convolution remains on PyTorch/cuDNN Conv2d.

## `compile_policy`

- `off`
- `shared`
- `reduce_overhead`
- `max_autotune`

Compilation caches are **shared by the whole VDN state**, not duplicated 50 times.
Gather, epilogue, feature preparation and the preallocated scan can therefore reuse the
same compiled callable across DiT blocks with identical static geometry.

Any compile failure is latched and falls back to the exact eager implementation instead
of being retried every block.

---

# LoRA / adapter execution

The default adapter path stays low-rank at runtime.

For an ordinary target:

```text
x -> A/down -> B/up -> add to native module output
```

No dense `B @ A` matrix is materialized.

For fused native H3 QKV, independent upstream Q/K/V terms are written into their exact
output slices.

When multiple adapter sets target the same module (for example Stage-B + turbo):

- all terms share **one down GEMM**;
- terms that target the same output slice share **one up GEMM**;
- scalar adapter strength/alpha is folded into the stored up matrix once.

For default+turbo fused QKV, the common case becomes approximately:

```text
1 shared down + 1 Q up + 1 K up + 1 V up
```

instead of evaluating every low-rank pair independently.

MiniMax-H3's fused `mlp.fc2` path does not call `fc2.forward`, so that target is kept on
ComfyUI's native factorized merge path.

### Independent adapter strength

Advanced inputs:

- `default_adapter_strength`
- `turbo_adapter_strength`

A value of `-1` inherits the global `strength`. Defaults therefore preserve the normal
released model while allowing controlled experiments without changing checkpoint
files.

---

# Attention backends

Available values:

| Backend | Meaning |
| --- | --- |
| `auto` | Calibration hit if available, then safe architecture/geometry heuristics |
| `grouped` | Exact grouped dense SDPA over runs of frames with identical window keys |
| `flex` | PyTorch FlexAttention using a cached block mask |
| `flash2` | Exact decomposed global/local attention with FA2 varlen for the local groups |
| `decomposed` | Blackwell FA4/CuTe varlen local groups + exact dense global/anchor rows |
| `reference` | Exact grouped SDPA with Comfy attention overrides disabled |
| `compat` | Grouped path routed through ComfyUI's normal attention dispatcher |

## Why FA2 is not blindly selected on a 4090

The varlen kernel can be fast, but decomposed attention also needs gather/scatter work.
On pre-Blackwell cards that overhead may cost more than the kernel saves for a specific
sequence/window geometry.

Therefore FA2 is:

- available explicitly;
- eligible when an explicit calibration proves it faster;
- not selected by a generic `auto` heuristic without that measurement.

## Blackwell / FA4

When `flash_attn.cute` is importable on a Blackwell-class CUDA device, `auto` may use the
FA4 decomposed path directly. If it fails at import or runtime, the failure is latched
and the model falls back to grouped SDPA.

---

# Per-GPU attention calibration

**Kirei VDN-H3 Calibrate Attention** benchmarks the exact backends on a synthetic
geometry and stores the fastest backend that also passes numerical parity against
grouped SDPA.

Calibration is explicit. A normal generation never launches hidden benchmarks.

The persistent key includes:

- PyTorch version;
- GPU name and compute capability;
- dtype;
- sequence length;
- heads/head dimension;
- video start/end;
- tokens per frame;
- number of latent frames;
- exact window bounds;
- anchor mode;
- number of grouped attention runs.

A calibration therefore cannot be accidentally reused for a different packed layout.

The default store is:

```text
ComfyUI/models/vdn/vdn_h3_calibration.json
```

The calibration node accepts packed global rows before the video (`global_tokens`) and,
when needed, `global_after` rows after the video segment.

---

# Experimental FP8 projection

FP8 is deliberately restricted to the branch's dominant `to_out_linear` projection.
Everything numerically sensitive remains BF16/FP32.

`auto`, `balanced`, `low_vram` and `max_speed` **never silently switch to FP8**.
Use either:

- `profile=experimental_fp8`, or
- advanced `projection_precision=fp8`.

Before constructing the FP8 branch, the node performs a small `_scaled_mm` capability
probe on the actual CUDA device. If the active GPU/PyTorch/CUDA path cannot execute the
required operation, the model falls back to BF16 **before rendering starts**.

### Architecture-dependent scaling

The implementation follows the upstream strategy:

- Blackwell-class path: per-tensor activation/weight scaling;
- earlier supported NVIDIA path: rowwise activation scaling and per-output-channel
  weight scaling.

The CPU master of an FP8 interior projection is actually stored as FP8 + FP32 scale, so
hybrid streaming reduces host storage, PCIe traffic and GPU staging size rather than
keeping a hidden BF16 duplicate.

### BF16 edge blocks

By default the first **4** and last **4** transformed blocks remain BF16, following the
upstream quality-oriented policy. `fp8_skip_end_blocks` exposes this count for controlled
experiments.

FP8 changes numerical execution and can change the final video trajectory. It is a
performance/quality experiment, not an exact-reference mode.

---

# Runtime diagnostics

Enable `diagnostics` only while profiling.

Timing scopes use CUDA events on the active stream; they do not synchronize before and
after every stage. The events are resolved together when the runtime report is
requested.

**Kirei VDN-H3 Runtime Report** returns JSON containing:

- branch placement;
- total/resident/streamed branch bytes;
- pinned host bytes;
- GPU stream-buffer bytes;
- H2D copy bytes/count;
- prefetch/request counters;
- requested and actual attention backend;
- backend failure latches;
- calibration file/count/last hit;
- kernel backend and compile policy;
- tile size;
- LoRA and curve factor memory by dtype/device;
- FP8 storage ratio/saved bytes when active;
- stage timings;
- CUDA allocated/reserved/peak memory.

To make the report execute after sampling, connect the sampler LATENT (or another
post-sampler value) to its optional `after` input.

---

# Validation tools

## CPU / dependency-light tests

```bash
pytest -q
```

The repository includes numerical tests for:

- VDN frame statistics and delta solve;
- bidirectional scan parity;
- gather parity;
- grouped attention versus dense-mask oracle;
- decomposed attention-plan coverage;
- factorized adapter mapping;
- shared runtime caches;
- tiled versus untiled branch execution;
- hybrid BF16/FP8 projection inventory;
- calibration persistence;
- exact-reference policy.

## General CUDA probe

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-gpu0.json
```

Checks temporal kernels, scan parity, attention and streaming.

## Consumer/workstation CUDA probe

```bash
python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic-gpu0.json
```

Checks:

- exact tiled branch versus untiled branch;
- hybrid double-buffer streaming correctness/telemetry;
- FP8 `_scaled_mm` availability, parity, storage and timing.

Use `--quick` for a smaller first pass.

The probes are intentionally separate from claims in this README: hardware-specific
performance must be measured in the actual ComfyUI environment.

---

# Advanced input reference

| Input | Default | Notes |
| --- | --- | --- |
| `branch_mode` | `auto` | `resident`, `hybrid`, `stream` |
| `lora_mode` | `auto` | `auto` currently prefers exact factorized bypass; `merge` is explicit |
| `attention_backend` | `auto` | grouped/Flex/FA2/FA4/reference/compat |
| `kernel_backend` | `auto` | temporal/feature kernel family |
| `compile_policy` | `auto` | compilation independent of kernel family |
| `tile_frames` | `0` | `0` means profile policy; explicit positive value overrides it |
| `pin_strategy` | `auto` | `all` can pin outside Comfy's normal budget; use deliberately |
| `projection_precision` | `auto` | `auto` stays BF16; FP8 is explicit/profile-only |
| `fp8_skip_end_blocks` | `4` | BF16 blocks at each end when FP8 is active |
| `default_adapter_strength` | `-1` | inherit global strength |
| `turbo_adapter_strength` | `-1` | inherit global strength |
| `strict_validation` | `true` | reject incompatible/incomplete checkpoint inventories early |
| `diagnostics` | `false` | enable event timing and extended memory report |

---

# Checkpoint placement

Place an authorized exploded VDN stage under a configured `models/vdn` root. The loader
expects the model specification, branch safetensors and declared adapter directories to
form a complete, self-consistent inventory; unknown weight files are not silently
ignored.

Pruned H3 AdaLN adapter grids can be packaged directly with the VDN checkpoint. The
runtime searches the checkpoint first, then configured VDN model roots, with legacy
sibling-node discovery only as a final compatibility fallback.

---

# MultiGPU

The current integration supports one compute device per patched H3 model.

ComfyUI `deepclone_multigpu` is rejected explicitly instead of copying runtime closures
and auxiliary state into a second GPU model incorrectly. Either GPU can still be chosen
as the H3 model's load device; the restriction is on splitting/cloning one patched model
across devices.

A true distributed/Ulysses VDN implementation would be a separate architecture, not a
safe automatic clone of this single-GPU runtime.

---

# What still requires a new trained checkpoint

Some attractive domestic optimizations cannot be created from the released weights by
runtime code alone:

- **64-dimensional VDN state**: would reduce state memory/solve cost substantially but
  changes the learned branch representation and requires retraining;
- **true 4-NFE turbo**: running an 8-step DMD adapter for only four steps is not the same
  model; a four-step distillation/consistency stage is required;
- **quantization-aware VDN training**: runtime FP8 can be tested today, but a checkpoint
  trained with the quantized projection in the loop is a different quality target.

See `docs/TRAINING_VARIANTS.md` for the proposed directions.

---

# Design documentation

- `docs/ARCHITECTURE.md` — exact dataflow, memory lifecycle and backend design.
- `docs/VALIDATION.md` — numerical/GPU validation matrix and profiling procedure.
- `docs/TRAINING_VARIANTS.md` — changes that require training rather than runtime code.

---

# License and attribution

Source in this repository is Apache-2.0. Adapted/informed upstream algorithms and
notices are recorded in `NOTICE` and `THIRD_PARTY.md`.

Model weights are not bundled. MiniMax-H3 and VDN-H3 weights are governed by their own
model license terms independently of this source-code license.
