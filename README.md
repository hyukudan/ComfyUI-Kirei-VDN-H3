# ComfyUI Kirei VDN-H3

Native, performance-oriented **VideoDeltaNet (VDN) integration for MiniMax-H3 in ComfyUI**.

Kirei patches the MiniMax-H3 model already loaded by ComfyUI. It keeps the native H3
loader, conditioning, audio/video packing, VAE, quantized model support and Comfy model
lifecycle, then adds the released VDN hybrid-attention branch and an optimized single-GPU
runtime around it.

The project has two goals that must coexist:

- preserve the released VDN-H3 architecture, adapters and sampling recipe;
- make the added VDN work efficient enough that it does not erase the speed of an
  already-optimized H3 base.

> Model weights are not included. Follow the licenses that apply to MiniMax-H3 and the
> VDN-H3 checkpoint you use.

---

## Quick start

### 1. Install a complete VDN stage

Keep the OpenVDN stage directory intact under `ComfyUI/models/vdn/`.

For Stage-DMD:

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

Kirei validates the stage specification, branch inventory, block/head geometry and
adapter targets. An incomplete checkpoint is rejected rather than partially applied.

### 2. Apply the released Stage-DMD model

Insert **Kirei Apply VDN-H3** after the H3 model loader.

```text
profile                  = auto
apply_turbo_adapter      = true
strength                 = 1.0
default_adapter_strength = -1   # inherit 1.0
turbo_adapter_strength   = -1   # inherit 1.0
```

The released Stage-DMD generator requires both adapters:

```text
default = 1.0
turbo   = 1.0
```

### 3. Use the correct denoising trajectory

For the released OpenVDN Stage-DMD checkpoint:

```text
sampler       = euler
scheduler     = simple
steps / NFE   = 8
denoise       = 1.0
video shift   = 12.0
audio shift   = 3.0
```

**Do not reuse `res_multistep` from a base-model workflow.** It changes the trajectory and
can produce hard or patterned output even when the VDN checkpoint itself is correct.

Current ComfyUI handles the MiniMax-H3 video/audio clocks in the model-sampling layer, so
stock Euler is valid for the canonical path.

---

# Released recipes

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

The Stage-DMD `turbo` adapter is not merely the external MiniMax-H3 Turbo LoRA attached
to VDN. OpenVDN initializes from that few-step adapter and then trains the student inside
the VDN manifold for the released eight-step trajectory.

## Stage-B — 50-NFE fidelity reference

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
profile         = reference
projection      = BF16
```

Use this first when investigating quality. If Stage-B/50 is already wrong, inspect the
VDN architecture, adapter mapping, window semantics or AdaLN handling before blaming the
Stage-DMD distillation adapter.

---

# Profiles

| Profile | Intended use | Main policy |
| --- | --- | --- |
| `auto` | Recommended | Quality-first adapters; exact calibrated local attention; placement/tiling from actual VRAM; **resident VDN projection stays BF16 until a quantized path is qualified**; hybrid/stream may follow a quantized base to reduce storage/H2D |
| `max_speed` | Explicit speed experiment | Resident branch, no tiling, adapter merge/requantization, aggressive projection precision, pointwise fusion, serial tuned execution |
| `balanced` | Manual general-purpose | Automatic placement/shared compilation with BF16 projection unless explicitly overridden |
| `low_vram` | 24 GB-class GPUs | Hybrid storage, exact five-frame tiling, factorized LoRA bypass; quantized projection may follow the base to reduce memory/transfer |
| `reference` | Numerical/fidelity oracle | BF16 projection, exact grouped SDPA, no external attention override, eager/reference-safe branch |
| `compat_reference` | Compatibility debugging | Reference VDN math while allowing Comfy's normal attention override path |
| `experimental_fp8` | Precision experiment | Explicit FP8 VDN projection with capability/fallback handling |
| `workstation_fp8` | Scheduling experiment | Resident FP8 branch, merged adapters and optional two-stream overlap |

A profile name is not a benchmark result. INT8, FP8, merge and parallel execution are
optimizations only after they beat the qualified control on the real GPU/geometry and
pass the visual gate.

## Why resident `auto` is BF16-first

A quantized H3 backbone does not imply that the new VDN output projection should also be
quantized automatically. On the current workstation measurements the VDN INT8 projection
did not beat BF16, and quantizing/merging adapters introduces extra numerical risk.

Therefore resident `auto` now uses:

```text
adapter mode       = bypass
VDN projection     = BF16
branch execution   = serial
local attention    = exact, calibrated
```

The benchmark matrix still contains explicit BF16, INT8 and FP8 VDN scenarios. Once a
quantized route demonstrably wins on a target GPU while preserving quality, it can be
promoted deliberately rather than assumed from the base storage format.

On a constrained hybrid/stream path, reducing projection size can be necessary for VRAM
and PCIe traffic; there `auto` may use the quantized base family while retaining adapter
bypass.

---

# What VideoDeltaNet changes

VDN-H3 is a trained hybrid of:

1. **local exact softmax attention** for short-range detail;
2. a **bidirectional recurrent linear branch** for context outside that local window.

Per latent frame:

```text
A_t = K_t^T diag(beta_t) K_t
B_t = V_t^T diag(beta_t) K_t
```

The released `vdn_solve` rule factorizes the SPD system around `I + A_t`, builds forward
and reverse state banks, and reads the state representing context outside the frame's
local softmax window.

## Preserved trained invariants

Kirei keeps checkpoint-declared values, including the released lineage:

- chunk-aligned windows;
- `chunk=5`, `radius=1`;
- first/last dense anchors (`both`);
- `vdn_solve`;
- alpha bridge;
- K/V short convolution;
- text-state initialization;
- independent softmax and linear gates;
- FP32 A/alpha/Cholesky/recurrent-state islands.

The released native port shares raw H3 Q/K/V with the branch. `linear_head_dim` must
therefore match the loaded H3 attention head dimension. A smaller state dimension needs a
separately trained checkpoint.

---

# Exact local attention is protected from quantized overrides

VDN local windows are trained/validated as exact attention. Normal Kirei paths therefore
do **not** route them through `optimized_attention_override`; Sage/kitchen quantized
attention cannot silently alter the VDN local window.

Exact does not mean slow. When available Kirei calls
`comfy.ops.scaled_dot_product_attention`, allowing current ComfyUI to prioritize the
fastest exact PyTorch backend for the platform:

```text
Flash -> cuDNN -> efficient -> math
```

This is especially useful on Windows builds while preserving the same attention model.

`compat_reference` is the explicit exception for experiments that intentionally want the
normal Comfy attention override chain.

## Exact acceleration backends

| Backend | Description |
| --- | --- |
| `grouped` | Exact SDPA over contiguous query runs sharing the same key set |
| `flex` | Exact FlexAttention with cached block mask |
| `flash2` | Dense global/anchor rows + FA2 varlen local groups |
| `decomposed` | Blackwell-oriented dense + FA4/CuTe varlen groups |
| `reference` | Exact grouped oracle, no external override |
| `compat` | Grouped compatibility path through normal Comfy attention dispatch |
| `auto` | Persistent exact winner for the current GPU/software/packed geometry |

Optional backend failures are latched. `auto` can qualify exact candidates against the
grouped oracle, persist the winner and reuse it for matching geometry.

---

# Mapping the tuned OpenVDN stack to ComfyUI

OpenVDN's optimized inference accelerates more than the recurrent branch. Kirei reuses
what current Comfy already does efficiently and implements the missing work natively.

| Tuned component | Provider |
| --- | --- |
| H3 fused QKV | ComfyUI MiniMax-H3 |
| Q/K RMSNorm + RoPE | ComfyUI / comfy-kitchen |
| H3 quantized wide linears | ComfyUI / comfy-kitchen |
| H3 SwiGLU-aware fused input path | ComfyUI |
| VDN window semantics | Kirei |
| grouped/Flex/FA2/FA4 exact dispatch | Kirei |
| temporal 5-tap conv + SiLU/L2Norm | Kirei Triton + exact fallbacks |
| frame-statistics prologue | Kirei shared runtime |
| scoped TF32 A GEMM | Kirei |
| Cholesky/transition factorization | Kirei |
| preallocated forward/reverse scan | Kirei |
| gather + RMSNorm/output-gate epilogue | Kirei shared compile cache |
| BF16/INT8/FP8 VDN output projection | Kirei |
| H3 pointwise pre/post fusion | Kirei on qualified resident paths |
| resident/hybrid/stream placement | Kirei + Comfy ModelPatcher |
| low-VRAM factorized adapters | Kirei |
| explicit speed merge/requantization | Comfy ModelPatcher |

---

# Adapter strategy

The Stage-B/Stage-DMD LoRA files are mapped to native H3 modules, including fused Q/K/V
slices, text-refiner targets, SwiGLU ordering, native `mlp.fc2` handling and curve AdaLN.

## Quality-first factorized bypass

`auto`, `balanced`, `low_vram` and reference-oriented paths preserve the low-rank update
in activation space.

All A/down terms targeting one native module share one down projection; terms that land
on the same output slice share one up projection. Default+turbo QKV therefore approaches:

```text
1 shared down
+ 1 Q up
+ 1 K up
+ 1 V up
```

without materializing dense `B @ A` deltas.

This is important on INT8/FP8/pruned bases: merging a small LoRA delta into an already
quantized weight can round away part of the update and soften the result.

## Explicit merge/requantization

`max_speed` / `workstation_fp8` can merge eligible adapters once at model load to remove
the runtime low-rank GEMMs. This route is faster only if the real benchmark says so and
must pass its own quality gate.

## Runtime recipe metadata

The Runtime Report records:

```text
adapters.active
adapters.strengths
adapters.lora_mode
adapters.reports
```

Canonical Stage-DMD benchmarking requires exactly:

```text
active    = [default, turbo]
strengths = {default: 1.0, turbo: 1.0}
```

---

# Pruned / curve AdaLN bases

Some H3 checkpoints collapse full AdaLN time projections onto a small shared curve basis.
Kirei detects this through either:

- `use_adaln_curves`, or
- the structurally small input width of the first block's `adaln_proj.linear`.

The structural test covers converted checkpoints whose flag is missing or unreliable.

Kirei does **not** silently skip the original LoRA's AdaLN terms. It reconstructs their
full time-conditioning input using the model's `adaln_t_table` and
`h3_silu_temb_grid.safetensors`, then applies the original low-rank update at runtime.

The e-grid search prefers checkpoint/model-root locations and uses the legacy
MiniMax-H3-Turbo sibling path only as a fallback. Missing required curve data is a clear
error, not a quality-degrading silent omission.

---

# VDN projection precision

The dominant branch GEMM is `to_out_linear`:

```text
projection_precision = auto | bf16 | int8 | fp8
```

### BF16

The reference and resident `auto` control. Use it first for fidelity and for the baseline
performance decision.

### INT8 / ConvRot

Uses current Comfy quantized-tensor/comfy-kitchen support when available. Sensitive
recurrent math stays BF16/FP32. INT8 is a Comfy-specific optimization candidate and is
not promoted merely because the H3 base is INT8.

### FP8

Explicit projection quantization with actual-device capability testing. A failed kernel
falls back before/during execution rather than crashing the render, but the benchmark
recorder refuses to file a BF16 fallback under an FP8 scenario label.

Quantized projection can change the denoising trajectory. Qualify final output, not only
local GEMM error.

---

# Linear branch execution and memory

## Tuned serial path

```text
native qkv_proj
   ├─ raw Q/K/V -> VDN branch -> projected linear delta
   └─ Q/K norm + RoPE -> local exact softmax -> H3 out_proj
                                           + linear delta
```

The branch consumes raw Q/K/V before native in-place Q/K normalization/RoPE. Once the
linear delta is projected, raw video Q/K/V do not remain alive through softmax.

## Exact tiled path

On constrained GPUs K/V are processed by frame tiles with the two-frame halo required by
the five-tap temporal convolution. Only compact A/B/alpha statistics remain between the
statistics pass and the global recurrence; Q/readout/projection are then processed by
tiles.

Tiling changes activation scheduling, not the VDN math.

## Scan

The tuned recurrence uses:

- batched transition/injection factorization;
- preallocated prefix/suffix banks;
- one `torch.baddbmm(..., out=...)` per frame/direction;
- no `.item()` synchronization in the loop.

The dependent scan is deliberately not compiled as one giant Inductor graph.

## Experimental parallel path

The softmax and recurrent branches are independent until their sum, so a large-VRAM
experiment can run the branch on a second CUDA stream. This also increases activation
lifetime and can cause SM contention. It is not assumed faster and remains outside normal
`auto`.

---

# Branch weight placement

### `resident`

All VDN branch tensors live in the Comfy model lifecycle on GPU. Preferred for a 96 GB
workstation when enough activation headroom remains.

### `hybrid`

Small VDN tensors stay resident while the dominant projection representation is streamed.
This is the normal consumer compromise.

### `stream`

CPU masters feed two reusable CUDA staging slots.

Streaming tracks allocated buffers separately from valid contents. Every slot carries its
current block identity and `valid_keys`; changing block invalidates content even if a
same-shaped buffer is still allocated. This prevents stale partial/gate-only loads from
being mistaken for valid full-block weights.

Prefetch uses reusable ready/consumed events, N+1 scheduling, cyclic last->first prefetch
and Comfy-aware pinned-memory policy.

---

# Shared runtime/cache policy

One patched `VDNState` owns caches shared by all 50 H3 blocks:

- gather indices;
- compiled static operations;
- temporal-kernel dispatch;
- attention masks/plans;
- exact backend calibration;
- failure latches.

Controls:

```text
kernel_backend = auto | triton | conv1d | eager
compile_policy = auto | off | shared | reduce_overhead | max_autotune
```

Compile/backend failures fall back and are remembered rather than retried in every block.

---

# H3 pointwise fusion

On qualified resident inference Kirei can compile shared bodies for:

```text
RMSNorm + AdaLN modulation
residual + gate * branch output
```

The compiled geometry is shared by all 50 blocks. A failure falls back to the native
Comfy block from the original residual; no intentionally slower eager imitation is kept
in the hot path.

---

# GPU policy

## RTX PRO 6000 Blackwell 96 GB

Normal `auto` target:

```text
branch             = resident when headroom permits
VDN projection     = BF16
adapter mode       = bypass
branch execution   = serial
attention          = exact calibrated winner
block fusion       = enabled when compile path qualifies
```

Explicit experiments:

- INT8/ConvRot projection;
- FP8 projection;
- `max_speed` merge/requantization;
- FA4;
- parallel branch execution.

## RTX 4090 24 GB

Normal targets:

- hybrid placement;
- five-frame exact tiles;
- adapter bypass;
- reduced projection storage/transfer when needed;
- grouped/Flex and FA2 when installed;
- stream as a lower-VRAM fallback.

---

# Recipe-aware benchmarks

Do not benchmark by changing titles or reusing saved workflow sampler widgets.

Kirei includes four nodes:

```text
Kirei Benchmark Scenario
Kirei Benchmark Sampling
Kirei Benchmark Start
Kirei Benchmark End
```

Recommended graph:

```text
Benchmark Scenario
      │ scenario_id
      ▼
Benchmark Sampling ◄──── MODEL
      ├── SAMPLER ────────────────┐
      ├── SIGMAS ─────────────────┤
      └── recipe_token            │
               │                  │
               ▼                  │
Benchmark Start ◄──────── MODEL   │
      │ MODEL                     │
      └───────────────────────────┼──► SamplerCustomAdvanced
                                 │
SamplerCustomAdvanced ◄──────────┘
      │ LATENT
      ▼
Benchmark End
```

**Benchmark Sampling creates the real Comfy SAMPLER and SIGMAS from
`benchmarks/scenarios.json`.** Benchmark Start requires its verified opaque token.

For Stage-DMD this mechanically enforces:

```text
Euler / simple / 8 NFE / denoise 1.0 / shifts 12,3
```

The result recorder then verifies:

- sampler/scheduler/NFE/denoise;
- actual H3 shifts;
- checkpoint `turbo_num_steps`;
- VDN profile;
- actual projection precision after fallback;
- active adapters and strengths.

A run using `res_multistep`, beta, six steps, wrong adapter strength or an INT8->BF16
fallback cannot be recorded under the canonical label.

## Active benchmark tiers

| Tier | Geometry | Purpose |
| --- | --- | --- |
| Quick | 608×352 / 121 requested frames | frequent regression |
| Detail | 960×544 / 121 | micro-detail and quantization quality |
| Primary | 608×352 / 241 | repeated performance decision |
| Stress | 608×352 / 401 | long-sequence scaling/crossover |
| Canonical quality | 1344×768 / 345 | Stage-B fidelity + Stage-DMD quality |

For the clean eight-NFE product control, Larry v4 runs at strength 1.0 with the same
Euler/simple/8 trajectory. The locally preferred hybrid/Sage/Larry-0.6/Euler-beta/6
workflow remains a separate visual reference rather than contaminating canonical speed
claims.

See [`benchmarks/README.md`](benchmarks/README.md) and
[`docs/VALIDATION.md`](docs/VALIDATION.md).

---

# Runtime Report

Connect **Kirei VDN-H3 Runtime Report** downstream of sampling when profiling.

It exposes:

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
attention_calibration / attention_failures
block_fusion / block_fusion_error
kernel_backend / compile_policy / tile_frames
lora_factors / curve_factors
diagnostics
CUDA memory
performance_analysis
```

Verify what actually executed before interpreting a timing number.

---

# CUDA probes

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-core-gpu0.json
python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic-gpu0.json
```

Run them independently on each target GPU. Synthetic probes qualify kernels; the final
quality/performance verdict belongs to the real Comfy generation workflow.

---

# Cold vs warm timing

The first render can include checkpoint parsing, model/adapter construction, projection
conversion, Triton/Inductor compile, attention calibration and allocator setup.

Standard protocol:

1. record cold separately;
2. warm at least once;
3. collect five warm runs by default;
4. primary metric = median sampler-only seconds;
5. record peak VRAM;
6. keep VAE decode/MP4 encoding as a separate end-to-end metric.

Never compare a warmed control with the first VDN compile/autotune run.

---

# When quality looks wrong

Check, in order:

1. Stage-DMD sampling is Euler/simple/8/denoise1;
2. `adapters.active = [default, turbo]` and strengths are both 1.0;
3. use BF16 projection before testing INT8/FP8;
4. confirm VDN local windows did not receive Sage/kitchen quantized override;
5. confirm required curve AdaLN factors were re-injected rather than skipped;
6. run Stage-B/50 reference to determine whether the problem predates Stage-DMD Turbo.

H3 can internally align requested output frames to its supported temporal grid. That by
itself is not a quality failure.

---

# When VDN looks slower

Treat it as a profiler result, not as an acceptable property of an optimization.

Check:

- cold vs warm;
- actual exact attention backend and calibration hit;
- linear branch vs softmax stage time;
- projection precision;
- adapter bypass vs merge;
- branch transfer waits;
- block-fusion fallback;
- resident/hybrid/stream placement;
- identical verified sampling trajectory between technical variants.

If INT8 is slower than BF16, keep BF16 until the path is improved. If VDN still loses
materially in the 401-frame stress workload, investigate the hot path rather than
explaining it away as a short-video crossover.

---

# Advanced inputs

| Input | Meaning |
| --- | --- |
| `branch_mode` | auto/resident/hybrid/stream |
| `branch_execution` | serial or experimental parallel |
| `lora_mode` | auto/bypass/merge |
| `attention_backend` | auto/grouped/flex/flash2/decomposed/reference/compat |
| `kernel_backend` | auto/triton/conv1d/eager |
| `compile_policy` | auto/off/shared/reduce_overhead/max_autotune |
| `tile_frames` | 0 lets the profile choose; positive forces exact tiling |
| `pin_strategy` | Comfy-aware pinning or explicit overrides |
| `projection_precision` | auto/bf16/int8/fp8 |
| `default_adapter_strength` | independent Stage-B strength |
| `turbo_adapter_strength` | independent Stage-DMD strength |
| `strict_validation` | reject incompatible inventory/geometry early |
| `diagnostics` | record CUDA-event stage telemetry |

Use `reference` rather than trying to rebuild the numerical oracle manually from
advanced switches.

---

# Compatibility and lifecycle

- VDN branch weights are Comfy additional models.
- LoRA bypass and curve factors participate in model lifecycle/accounting.
- applying VDN twice is rejected;
- patch collisions are rejected rather than silently overwritten;
- runtime state represents one selected compute GPU; distributed/Ulysses requires real
  sharding/communication logic;
- **Kirei Release VDN-H3 Weights** releases VDN auxiliaries/caches;
- the legacy node id remains for old saved workflows.

---

# Upstream reference

VDN mathematics, released checkpoint architecture and the tuned inference design come
from **OpenVDN / VDN-Minimax-H3**:

- `https://github.com/OpenVDN/vdn-minimax-h3`
- `https://openvdn.github.io/`

Kirei's Comfy-specific work focuses on native model patching, exact attention dispatch,
quantized H3 compatibility, model-memory lifecycle, profiling and workstation/consumer
execution.

See `THIRD_PARTY.md` and `NOTICE` for attribution details.

---

# Reproducible performance claims

Record at least:

```text
GPU + VRAM
CUDA / PyTorch / ComfyUI revision
H3 base checkpoint + storage precision
VDN checkpoint
active adapters + strengths + lora mode
sampler + scheduler + NFE + denoise
video/audio shifts
resolution + requested frames
actual packed/latent geometry
VDN projection precision
attention backend + calibration hit
branch placement + execution mode
block fusion status
cold time
warm median sampler time
peak VRAM
quality status
```

The benchmark suite exists so these conditions are machine-checkable rather than inferred
from workflow names or remembered widget values.

---

## License

Repository code follows the license declared in this repository. Adapted upstream
algorithms retain their required notices and attribution. Model weights are distributed
separately and may be subject to different terms.
