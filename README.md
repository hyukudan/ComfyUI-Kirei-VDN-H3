# ComfyUI Kirei VDN-H3

Native, performance-oriented **VideoDeltaNet (VDN) integration for MiniMax-H3 in ComfyUI**.

This custom node patches the MiniMax-H3 model that ComfyUI already loaded. It does not
replace the H3 loader, sampler, VAE, audio stack, conditioning pipeline, quantized model
support, or ComfyUI's model-management system.

The integration has two goals that must coexist:

- preserve the released VDN-H3 architecture and its exact BF16/FP32 reference path;
- make the inference path behave like an optimized ComfyUI model rather than a research
  reference implementation bolted onto a fast H3 base.

That distinction matters. A fast INT8/ConvRot H3 base followed by a new BF16 VDN
projection, runtime LoRA GEMMs and unfused block pointwise work can be **slower** even
though VDN reduces the attention complexity. The current runtime therefore matches the
VDN branch to the precision and execution family of the loaded H3 model whenever it is
safe to do so.

> Model weights are not included in this repository. Use only checkpoints you are
> authorized to use and follow the licenses that apply to MiniMax-H3 and the VDN-H3
> checkpoint.

---

## Quick start

Place a complete VDN checkpoint under a ComfyUI VDN model directory and restart or
refresh ComfyUI. Then insert **Kirei Apply VDN-H3** between the MiniMax-H3 model loader
and the sampler.

For the normal case:

```text
profile = auto
apply_turbo_adapter = true
strength = 1.0
```

Leave the advanced fields at `auto` unless you are profiling a specific path.

The node validates the checkpoint inventory, H3 block count, head geometry and adapter
targets before installing the runtime patch. A partially applicable checkpoint is
rejected instead of silently producing a hybrid of incompatible pieces.

---

# Profiles

| Profile | Intended use | Main policy |
| --- | --- | --- |
| `auto` | Recommended | Matches the loaded H3 precision family, chooses branch placement from real VRAM, uses the qualified serial tuned scheduler, merges LoRA once when a quantized model is fully resident, tiles on consumer GPUs, fuses H3 pointwise work when safe, and autotunes exact attention backends per geometry |
| `max_speed` | Large-VRAM benchmark / workstation | Resident branch, no tiling, LoRA merge, native quantized VDN projection when available, `reduce-overhead` compilation, H3 pointwise fusion, serial tuned scheduler |
| `balanced` | Manual general-purpose profile | Automatic branch placement and shared static compilation without forcing a precision family that the base does not use |
| `low_vram` | Consumer / 24 GB-class GPUs | Hybrid branch storage, 5-frame exact tiles, factorized LoRA bypass, native-precision VDN projection when available |
| `reference` | Numerical oracle | BF16/FP32 VDN path, exact SDPA, no Comfy attention override, eager/autograd-safe branch, no quantized VDN projection |
| `compat_reference` | Compatibility debugging | Reference VDN math while allowing ComfyUI's normal attention dispatch |
| `experimental_fp8` | Explicit precision experiment | Forces the large VDN projection to FP8 when the active runtime passes the capability probe |
| `workstation_fp8` | Experimental large-VRAM scheduling | Resident FP8 branch with the optional two-stream branch/softmax overlap when the GPU has enough headroom |

## What `auto` means on a quantized H3 base

`auto` detects the actual storage family of the loaded H3 QKV weight.

### TensorWise INT8 + ConvRot base

The expected tuned path is:

```text
H3 wide linears        -> ComfyUI TensorWiseINT8 / ConvRot
VDN to_out_linear      -> TensorWiseINT8 / ConvRot
VDN recurrent state    -> FP32 where trained/required
VDN narrow weights     -> BF16/FP32
Stage-B + Turbo LoRA   -> merged/requantized once when the model is resident
H3 MLP                 -> native Comfy linear_input_act INT8 path
H3 pointwise block     -> compiled fused pre/post kernels when qualified
attention              -> exact backend selected by persistent geometry autotune
scheduler               -> serial tuned single-GPU path
```

This is particularly important in ComfyUI because an INT8 H3 base is already very fast.
Adding fifty large BF16 VDN output projections on top of it defeats a large part of the
speed advantage.

### FP8 base

`auto` keeps the VDN output projection in the FP8 family when the runtime supports the
required scaled matrix multiplication.

### BF16 base

`auto` keeps the VDN output projection in BF16. Precision-changing modes remain
explicit unless the base model itself already establishes that quantized inference
family.

---

# Expected policy by GPU class

## RTX PRO 6000 Blackwell 96 GB

With enough free VRAM, `auto` normally uses:

- resident VDN branch weights;
- no activation tiling;
- native base precision for `to_out_linear` — INT8/ConvRot when the H3 base is INT8;
- LoRA merge/requantization at model load instead of runtime low-rank GEMMs;
- compiled H3 pre/post pointwise fusion;
- serial tuned scheduling by default;
- persistent exact attention autotuning across grouped SDPA, Flex, FA2 and FA4 when the
  corresponding backend is installed.

`branch_execution=parallel` remains available as an advanced experiment. It trades
additional compact raw Q/K/V copies for overlap between the linear branch and local
softmax. It is **not** selected by `auto` until real-device benchmarks justify making it
the default.

## RTX 4090 24 GB

`auto` normally prefers:

- `hybrid` branch placement;
- 5-frame exact tiles for the VDN linear path;
- the loaded H3 precision family for the large VDN projection;
- factorized LoRA bypass rather than a low-VRAM dynamic quantized merge;
- streamed `to_out_linear` with the smaller gates/convolutions/recurrent weights kept
  resident when memory allows;
- exact attention backend chosen from a persistent calibration result when more than one
  accelerated backend is available.

The goal on a 24 GB card is not to imitate the 96 GB execution plan. It is to keep the
large model usable without turning PCIe traffic or activation lifetime into the new
bottleneck.

These are runtime policies, not benchmark claims. Always validate on the actual ComfyUI
build, PyTorch/CUDA version, GPU and render geometry.

---

# What VideoDeltaNet changes

VDN-H3 replaces one dense attention path with a hybrid of:

1. **local softmax attention** for short-range detail;
2. a **bidirectional recurrent linear branch** for context outside that local window.

For each latent video frame, the linear branch constructs compact frame statistics:

```text
A_t = K_t^T diag(beta_t) K_t
B_t = V_t^T diag(beta_t) K_t
```

The released VDN rule then forms the frame transition around:

```text
(I + A_t)^-1
```

and runs one recurrence forward in time and another backward in time. The readout for a
frame uses state from outside the same frame's local softmax coverage, so the two
branches are complementary rather than double-counting the same neighborhood.

## Trained invariants kept by this implementation

The runtime preserves the checkpoint's architecture:

- chunk-aligned local window;
- released `chunk=5`, `radius=1` behavior when declared by the checkpoint;
- first/last dense anchors in the trained row/column mode;
- `vdn_solve` transition;
- alpha bridge;
- optional text-state initialization in both recurrent directions;
- FP32 state statistics and recurrence where required;
- Cholesky-based solve;
- softmax and linear output gates;
- trained K/V short convolutions;
- strict block/head/shape validation.

The released H3 integration shares raw native Q/K/V with the VDN branch. Therefore
`linear_head_dim` must match the loaded H3 attention `head_dim`. A 64-dimensional VDN
state would require a separately trained checkpoint; it is not a runtime optimization
switch.

---

# Mapping the upstream tuned stack to ComfyUI

The optimized OpenVDN inference path accelerates more than the new recurrent branch.
Some of those optimizations are already provided by current ComfyUI, while others are
implemented by this node.

| Tuned component | Provider in this integration |
| --- | --- |
| Fused native QKV projection | ComfyUI MiniMax-H3 |
| Fused Q/K RMSNorm + RoPE | ComfyUI `comfy.quant_ops.ck.rms_rope_split_half_` |
| INT8 H3 wide linears / ConvRot | ComfyUI / comfy-kitchen |
| H3 MLP `fc2` with SwiGLU-aware input path | ComfyUI `linear_input_act` |
| Local VDN attention mask/window | Kirei VDN-H3 |
| Grouped / Flex / FA2 / FA4 exact dispatch | Kirei VDN-H3 |
| Softmax gate + output repack | Kirei VDN-H3 compiled fusion |
| VDN temporal 5-tap conv + SiLU + L2Norm | Kirei Triton kernel with fallbacks |
| Frame-statistics preparation | Kirei shared compiled prologue |
| Scoped TF32 for the A GEMM only | Kirei VDN-H3 |
| Batched Cholesky / transition factorization | Kirei VDN-H3 |
| Preallocated forward/reverse `baddbmm(out=...)` scan | Kirei VDN-H3 |
| Gather + RMSNorm/output gate epilogue | Kirei shared compiled kernels |
| Large VDN `to_out_linear` quantization | Kirei native INT8/ConvRot or FP8 projection |
| H3 RMSNorm + AdaLN pre-stage fusion | Kirei compiled block kernel when resident |
| H3 gated residual post-stage fusion | Kirei compiled block kernel when resident |
| LoRA merged before hot inference | ComfyUI ModelPatcher on qualified resident path |
| Low-VRAM LoRA without dense B@A | Kirei factorized bypass |

The guiding rule is simple: **do not replace a ComfyUI optimization that is already
better integrated with model management; add only the tuned work that Comfy does not
already perform.**

---

# Native-precision VDN output projection

`to_out_linear` is the dominant VDN branch weight and one of its largest per-token GEMMs.
The runtime exposes:

```text
projection_precision = auto | bf16 | int8 | fp8
```

## INT8 / ConvRot

When the H3 base uses `TensorWiseINT8Layout`, `auto` quantizes the VDN projection with
the same current ComfyUI API:

```python
QuantizedTensor.from_float(
    weight,
    "TensorWiseINT8Layout",
    per_channel=True,
    convrot=True,
    convrot_groupsize=256,
)
```

The forward path uses `comfy.quant_ops.ck.int8_linear`. The recurrent state, Cholesky,
alpha and other precision-sensitive components are **not** converted to INT8.

Quantized projection storage is real: the CPU master contains INT8 data plus FP32 scale,
not a hidden BF16 duplicate. Hybrid streaming therefore reduces host memory and PCIe
bytes as well as the GPU GEMM cost.

## FP8

The FP8 implementation is restricted to the large VDN projection and follows the
architecture-dependent scaling strategy used by the upstream inference work.

A small capability probe runs before the model state is installed. If the active
GPU/PyTorch/CUDA combination cannot execute the required path, projection construction
falls back to BF16 before sampling.

FP8 changes numerical execution. It can move a denoising trajectory to a different
sample even when one-step prediction remains close. Use `reference` when reproducibility
against BF16 is the goal.

---

# LoRA strategy

VDN-H3 ships adapter terms in addition to the recurrent branch. The best execution
strategy depends on model residency.

## Resident quantized model: merge once

When `auto` sees a quantized H3 base and the model is on the resident workstation path,
it uses ComfyUI's normal ModelPatcher to apply the LoRA during model loading. For
quantized weights Comfy dequantizes/applies the patch and calls the tensor's `set_func`
to rebuild the storage representation.

The result is that the hot forward continues to use the native quantized linear instead
of paying extra low-rank GEMMs in every one of the 50 blocks.

## Hybrid / low-VRAM: factorized bypass

When model management may move weights dynamically, the runtime keeps LoRA factors
compact:

```text
x -> A/down -> B/up -> add to native output
```

For a fused QKV target, multiple default/turbo terms share one down projection and one
up projection per output slice. A common default+turbo QKV case is therefore reduced to
approximately:

```text
1 shared down + 1 Q up + 1 K up + 1 V up
```

No dense `B @ A` delta is materialized.

`mlp.fc2` remains on ComfyUI's native patch path because MiniMax-H3 consumes it through
`linear_input_act` rather than through `fc2.forward`.

### Independent strengths

Advanced inputs:

- `default_adapter_strength`
- `turbo_adapter_strength`

`-1` means “inherit the global `strength`”.

---

# H3 pointwise block fusion

Current ComfyUI already performs the per-segment AdaLN scale/shift and residual gate
in-place, which is much better than a naive implementation. The upstream tuned VDN path
goes one step further for a fixed inference geometry: it compiles the full

```text
RMSNorm -> AdaLN scale/shift
```

and

```text
residual + gate * branch_output
```

stretches so that gathered modulation rows and intermediate tensors stay in registers.

Kirei VDN-H3 ports that strategy for **resident, inference-only** execution. The pre and
post bodies are compiled once per static geometry and reused by all 50 transformer
blocks.

Safety behavior:

- scalar and per-token modulation row layouts are supported;
- no GPU `.item()` is used to construct per-token modulation maps;
- if compilation is unavailable or a shape fails, the optimization is permanently
  latched off for that patched model;
- fallback restarts the whole block from the original residual and uses native ComfyUI
  execution;
- there is no intentionally slower eager imitation of the “fused” path.

The Runtime Report exposes `block_fusion` and `block_fusion_error`.

---

# Linear-branch execution and memory

## Low-activation serial path

The qualified default scheduler evaluates the VDN branch before Q/K are modified by
native RMSNorm+RoPE:

```text
fused native QKV
    |
    +-- raw Q/K/V views -> VDN linear branch -> projected VDN delta
    |
    +-- Q/K norm + RoPE -> local softmax -> H3 out_proj
                                             |
                                             +-- add VDN delta
```

This avoids retaining three compact raw-video copies during softmax. It is the default
for `auto` and `max_speed`.

## Optional parallel workstation path

`branch_execution=parallel` copies compact raw Q/K/V and runs the VDN branch on a
separate CUDA stream while the main stream performs local softmax. The streams join only
at the trained branch sum.

This can only help when:

- branch weights are resident;
- the GPU has enough spare VRAM for the compact raw copies;
- the two kernel mixes actually overlap well on that GPU.

Because concurrency is hardware- and shape-dependent, it is **not automatically used by
`auto` or `max_speed`**. `workstation_fp8` and the advanced override exist for controlled
benchmarking.

---

# Exact tiled execution for consumer GPUs

`tile_frames > 0` reduces live activation memory without changing the VDN recurrence.

### Pass 1: K/V statistics

For each tile, with the exact temporal-convolution halo:

```text
raw K/V
 -> spatial depthwise Conv2d
 -> temporal 5-tap conv
 -> SiLU / L2Norm where trained
 -> beta
 -> A_t / B_t
```

Only compact per-frame statistics remain live.

### Global recurrence

All frame transitions are factorized in batch. The forward and reverse recurrent banks
are preallocated and filled with one `torch.baddbmm(..., out=...)` launch per frame and
direction.

The recurrence loop itself is **not** handed to `torch.compile` as one huge dependent
graph. This follows the tuned upstream strategy and avoids unnecessary first-run and
re-specialization cost.

### Pass 2: Q/readout/output projection

```text
raw Q
 -> feature map
 -> gather outside-window recurrent state
 -> readout
 -> RMSNorm/output gate
 -> quantized or BF16 to_out_linear
 -> write final VDN delta tile
```

Dense first/last anchor rows are not sent through the dominant projection when their VDN
contribution is known to be zero.

`low_vram` defaults to five latent frames per tile.

---

# Branch weight placement

## `resident`

Every VDN branch tensor is a Comfy-managed auxiliary model on the GPU. This is the
preferred 96 GB path.

## `stream`

All branch tensors have CPU masters and are copied through two reusable CUDA staging
slots.

## `hybrid`

Small branch weights remain resident while the dominant projection is streamed. This is
normally the useful 24 GB compromise.

Streaming includes:

- explicit per-slot valid-key tracking;
- two reusable buffers;
- reusable `ready` / `consumed` CUDA events;
- a dedicated prefetch stream;
- block N+1 prefetch while block N computes;
- cyclic last-block -> first-block prefetch for the next denoising step;
- ComfyUI's pinned-memory budget by default;
- H2D byte/copy counters in runtime telemetry.

`pin_strategy=all` is an explicit aggressive override. Normal `auto` does not bypass
ComfyUI when it refuses additional pinned host memory.

---

# Attention backends

| Backend | Description |
| --- | --- |
| `grouped` | Exact grouped SDPA over runs with identical local-window keys |
| `flex` | Exact PyTorch FlexAttention with cached block mask |
| `flash2` | Exact global/local decomposition using FA2 varlen for local groups |
| `decomposed` | Exact Blackwell-oriented FA4/CuTe varlen local groups plus dense global/anchor rows |
| `reference` | Exact grouped SDPA with Comfy attention override disabled |
| `compat` | Grouped attention through ComfyUI's normal attention dispatcher |
| `auto` | Persistent winner for the exact GPU/PyTorch/geometry; otherwise performs a one-time exact backend autotune when several accelerated choices are installed |

## Runtime autotune

A fixed rule such as “few window groups means grouped SDPA is fastest” is not reliable
across a 4090, Blackwell, different PyTorch builds and different packed sequence lengths.

For an unseen geometry, `auto` can:

1. warm each installed exact candidate once;
2. time one steady execution;
3. compare output against grouped SDPA within the configured tolerance;
4. persist the fastest valid backend;
5. reuse that result on later renders with the exact same signature.

The signature includes:

- PyTorch version;
- GPU name and compute capability;
- dtype;
- sequence/head geometry;
- latent frame count;
- video start/end;
- tokens per frame;
- exact frame-window bounds;
- anchor mode.

The default store is:

```text
ComfyUI/models/vdn/vdn_h3_calibration.json
```

The first render can therefore be materially slower than steady state. Use **Kirei
VDN-H3 Calibrate Attention** when you want a deliberate multi-run benchmark instead of
the lightweight first-use autotune.

A backend import or runtime failure is latched, so a broken optional acceleration is not
retried in every block.

---

# VDN linear kernels

## Feature path

The five-tap temporal short convolution has a fused Triton implementation:

```text
temporal depthwise conv -> SiLU -> optional L2Norm
```

with exact fallbacks through compiled/eager Conv1d paths.

The spatial 5x5/declared depthwise convolution remains on cuDNN Conv2d, which is already
the appropriate kernel family.

Q/K/V feature output can be written directly as:

```text
[frames, heads, tokens_per_frame, head_dim]
```

so the frame-statistics GEMMs do not require an extra full-feature repack.

## Frame-statistics prologue

The runtime shares one compiled preparation body across all transformer blocks. It:

- makes K contiguous;
- forms K in FP32;
- applies beta weighting;
- prepares weighted V;
- computes A in FP32;
- scopes TF32 only around the A GEMM during inference;
- symmetrizes A;
- computes B and promotes the recurrent result to FP32.

No autocast surrounds the Cholesky solve or the FP32 recurrence.

## Scan

The tuned scan keeps:

- batched transition/injection factorization;
- preallocated prefix/suffix banks;
- one `baddbmm(out=...)` per latent frame and direction;
- no `.item()` synchronization in the hot path.

## Readout / epilogue

Gather indices are cached by exact window geometry. RMSNorm + output gate + final layout
repack are compiled as a shared epilogue.

---

# Compilation policy

`kernel_backend` and `compile_policy` are separate controls.

## `kernel_backend`

```text
auto | triton | conv1d | eager
```

## `compile_policy`

```text
auto | off | shared | reduce_overhead | max_autotune
```

Compiled function caches belong to one patched VDN state and are shared across all 50
H3 blocks. A compile failure is remembered and falls back instead of repeatedly
recompiling the same unsupported shape.

`max_speed` uses `reduce_overhead` on the qualified serial path. The experimental
parallel scheduler avoids forcing CUDA graph capture across two independent model
streams.

---

# Understanding first-run versus steady-state timing

Do not compare the first VDN render with a warmed Turbo render.

A first-use VDN path may include:

- checkpoint parsing;
- auxiliary model construction;
- LoRA merge/requantization;
- INT8 or FP8 VDN projection conversion;
- Triton compilation;
- Inductor compilation;
- attention backend warmup/autotuning;
- allocator setup.

Steady-state benchmarking should use the same:

- prompt/conditioning;
- output geometry;
- frame count;
- sampler and NFE count;
- H3 base precision;
- VAE/audio settings;
- CUDA/PyTorch build;
- warmed state.

The upstream published VDN results report steady denoising performance and exclude model
loading, warm-up, VAE decode and MP4 encoding. Follow the same separation when comparing
this integration.

---

# When VDN appears slower than conventional H3/Turbo

Treat this as a profiler problem, not as an expected property of an “optimization”.

Enable `diagnostics`, connect **Kirei VDN-H3 Runtime Report** after the sampler, and check
these fields first:

```text
base_precision
projection.precision
branch_mode
branch_execution
block_fusion
block_fusion_error
attention_last
attention_calibration.last_hit
lora_factors.bytes
branch_storage.h2d_bytes
performance_analysis
```

## On a resident INT8/ConvRot workstation path

The expected report is approximately:

```text
base_precision        = int8
projection.precision  = int8
branch_mode           = resident
branch_execution      = serial
block_fusion          = true
lora_factors.bytes    = 0   # or no bypass factors when all eligible LoRA merged
attention_last        = calibrated exact winner
```

If `base_precision=int8` but `projection.precision=bf16`, the INT8 VDN projection probe
failed or fell back. That should be investigated before interpreting the benchmark.

If `block_fusion=false`, inspect `block_fusion_error`.

If `attention_last=grouped` and there is no calibration hit while other accelerated
backends are installed, the attention backend has not yet reached a stable tuned choice.

If H2D transfer time is visible on a 96 GB card, the branch is not actually staying
resident.

---

# Runtime diagnostics

`diagnostics=true` uses CUDA events recorded on the active stream rather than
synchronizing before and after every scope. Events are resolved when the Runtime Report
is requested.

The report includes:

- selected profile and detected H3 base precision;
- actual VDN projection precision;
- branch placement and execution scheduler;
- resident / streamed / pinned bytes;
- H2D copy bytes and request counts;
- last real packed H3 geometry;
- attention backend, failure latches and calibration hit;
- block fusion status and fallback reason;
- kernel/compile/tile policy;
- LoRA/curve auxiliary storage;
- stage timings;
- CUDA allocated/reserved/peak memory;
- a compact bottleneck analysis and recommendations.

CUDA scopes on different streams may overlap. Their totals are useful for locating a
bottleneck but must not be blindly added to estimate wall-clock time.

To force the report to run after sampling, connect the sampler LATENT or another
downstream value to its optional `after` input.

---

# Validation

## Dependency-light suite

```bash
pytest -q
```

The repository covers, among other things:

- VDN solve and frame statistics;
- reference/inference scan parity;
- gather/window parity;
- grouped attention versus dense-mask oracle;
- decomposed-plan exact coverage;
- FA2 symbol/import regression;
- factorized QKV LoRA slicing;
- grouped adapter GEMMs;
- tiled versus untiled branch parity including temporal halo;
- shared runtime caches;
- hybrid projection inventory;
- FP8 and INT8 storage dtype preservation;
- Comfy `TensorWiseINT8Layout` detection;
- modulation segments with scalar and per-token row indices;
- reference profile rejecting quantized VDN execution;
- workstation versus low-VRAM runtime policy.

## CUDA probes

General optimized-runtime probe:

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-gpu0.json
```

Consumer/workstation probe:

```bash
python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic-gpu0.json
```

These probes are intentionally separate from performance claims. The final validation
must happen in the real ComfyUI environment because Triton, comfy-kitchen, FA2/FA4,
Inductor and quantized layouts are runtime-dependent.

---

# Advanced input reference

| Input | Default | Description |
| --- | --- | --- |
| `branch_mode` | `auto` | `resident`, `hybrid`, `stream`; controls VDN branch weight placement |
| `branch_execution` | `auto` | `serial` tuned default or explicit experimental `parallel` |
| `lora_mode` | `auto` | Resident quantized speed path merges/requantizes once; low-VRAM path uses factorized bypass |
| `attention_backend` | `auto` | `grouped`, `flex`, `flash2`, `decomposed`, `reference`, `compat` |
| `kernel_backend` | `auto` | `triton`, `conv1d`, `eager` overrides for linear feature kernels |
| `compile_policy` | `auto` | `off`, `shared`, `reduce_overhead`, `max_autotune` |
| `tile_frames` | `0` | `0` lets the profile decide; positive value forces exact frame tiling |
| `pin_strategy` | `auto` | `auto/comfy` respects Comfy's budget; `all` is an aggressive host-RAM override; `none` disables pinning |
| `projection_precision` | `auto` | `bf16`, `int8`, `fp8`; `auto` follows a quantized H3 base |
| `default_adapter_strength` | `-1` | Inherit global strength when `-1` |
| `turbo_adapter_strength` | `-1` | Inherit global strength when `-1` |
| `strict_validation` | `true` | Reject checkpoint/base geometry or inventory mismatch early |
| `diagnostics` | `false` | Record CUDA-event timing and runtime telemetry |

Use `reference` rather than manually combining advanced switches when numerical oracle
behavior is required.

---

# Optional acceleration dependencies

The runtime is designed with exact fallbacks. Optional packages improve particular
paths but their absence must not make the node unusable.

- **Triton**: fused temporal VDN kernel and other compiled CUDA paths;
- **comfy-kitchen**: current ComfyUI INT8/ConvRot execution family;
- **PyTorch FlexAttention**: exact block-mask attention option;
- **FlashAttention 2**: optional varlen local attention on supported pre-Blackwell GPUs;
- **FlashAttention 4 / CuTeDSL**: optional Blackwell-oriented decomposed attention.

An optional backend that fails is latched out and the runtime falls back to another exact
path.

---

# Compatibility and lifecycle

- VDN auxiliary weights are registered through ComfyUI additional `ModelPatcher`
  instances for memory accounting and offload lifecycle.
- Applying VDN twice to the same MODEL is rejected.
- Existing attention object-patch collisions are rejected instead of overwritten.
- MultiGPU deep-cloning is guarded until the distributed/Ulysses VDN execution path is
  explicitly ported.
- The legacy node identifier remains registered so saved workflows from earlier
  versions can still open.
- **Kirei Release VDN-H3 Weights** releases branch caches, quantized staging buffers,
  attention plans, curve factors and runtime compilation state associated with the
  patched model.

---

# Upstream reference

This integration is based on the architecture and released inference implementation of
**OpenVDN / VDN-Minimax-H3**:

- project: `https://github.com/OpenVDN/vdn-minimax-h3`
- technical project page: `https://openvdn.github.io/`

The official inference stack is the reference for VDN mathematics and for the tuned
single-GPU design. ComfyUI-specific changes in this repository focus on preserving
Comfy's quantized model formats, memory management, patch lifecycle and already-fused
MiniMax-H3 operations.

See `THIRD_PARTY.md` and `NOTICE` for attribution details.

---

# Performance claims

Do not infer a speedup from the existence of a backend or a profile name.

A performance claim for this integration should record at least:

```text
GPU + VRAM
CUDA / PyTorch / ComfyUI revision
H3 base precision/layout
VDN projection precision
attention backend + calibration hit
branch mode + execution mode
block fusion status
resolution
output frame count
latent frame count
NFE / sampler
first-run time
steady-state sampler time
peak VRAM
```

The objective of `auto` is to select a technically appropriate tuned path. The objective
of the benchmark suite is to prove that choice on the real machine and to expose the
exact stage responsible whenever it does not win.

---

## License

Repository code follows the license declared in this repository. Adapted upstream
algorithms retain their required notices and attribution. Model weights are distributed
separately and may be subject to different terms.
