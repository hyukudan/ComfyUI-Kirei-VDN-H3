# VDN-H3 validation and profiling

This document defines the qualification protocol for Kirei VDN-H3. It separates four
questions that are easy to mix accidentally:

1. is the VDN mathematics correct?;
2. is the released checkpoint being applied completely?;
3. is the model being sampled with the trajectory it was trained for?;
4. is an optimized runtime actually faster on the target GPU without failing the quality gate?

A valid result needs all four.

---

## 1. Canonical inference trajectories

Sampler, scheduler, NFE, denoise, CFG and MiniMax-H3 shifts are part of model identity
for a benchmark. Resolution and frame count alone are not enough. OpenVDN renders the
distilled model with a single model evaluation per step, so CFG must stay at 1.0; the
result recorder cannot see the guider, so this one is on the operator.

### OpenVDN Stage-DMD release

Canonical few-step VDN:

```text
checkpoint      = stage-dmd-step-250
adapters        = default 1.0 + turbo 1.0
sampler         = euler
scheduler       = simple
steps / NFE     = 8
denoise         = 1.0
cfg             = 1.0
video shift     = 12.0
audio shift     = 3.0
```

`res_multistep` is **not** valid for this recipe. It belongs to a base-model workflow and
changes the denoising trajectory. A visually plausible output produced with the wrong
sampler is not evidence about VDN quality.

### OpenVDN Stage-B fidelity reference

Use this before blaming the DMD/Turbo adapter when quality is suspicious:

```text
checkpoint      = stage-b-step-2000
adapters        = default 1.0
turbo           = OFF
sampler         = euler
scheduler       = simple
steps / NFE     = 50
denoise         = 1.0
cfg             = 1.0
video shift     = 12.0
audio shift     = 3.0
projection      = BF16
profile         = reference
```

If this path already diverges visually, investigate the VDN architecture, Stage-B
adapter conversion, window semantics or AdaLN handling before examining Stage-DMD.

### Larry MiniMax-H3 Turbo v4 clean control

For a clean same-trajectory quality/control comparison with Stage-DMD:

```text
LoRA            = Larry v4
strength        = 1.0
sampler         = euler
scheduler       = simple
steps           = 8
denoise         = 1.0
cfg             = 1.0
video shift     = 12.0
audio shift     = 3.0
```

Larry v4 supports 4–8 steps, while 6–8 is useful when motion/detail at the four-step
minimum is not sufficient. On current ComfyUI, stock Euler is valid because the H3
video/audio dual schedule is handled by the model sampling layer.

### Local visual workflow reference

The locally observed workflow that looked good is kept as a separate non-comparable
reference:

```text
hybrid blocks   = b30-49
SageAttention   = enabled
Larry strength  = 0.6
sampler         = euler
scheduler       = beta
steps           = 6
```

It is useful for visual context. It is not the canonical OpenVDN or clean Larry control.

---

## 2. Benchmark graph must own sampler and sigmas

Do not reuse sampler widgets from a production/smoke-test workflow.

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

`Kirei Benchmark Sampling` constructs the actual Comfy sampler and sigma schedule from
`benchmarks/scenarios.json`. `Kirei Benchmark Start` refuses to run without the opaque
verified recipe token produced by that node.

For Stage-DMD this guarantees:

```text
Euler / simple / 8 / denoise 1.0 / shifts 12,3
```

A manually configured `res_multistep`, `beta`, six-step or partial-denoise path cannot be
recorded as the canonical scenario.

---

## 3. Result recorder validation

`benchmarks/record_result.py` validates the measurement before appending it to the
result set.

For VDN it requires:

- verified sampling plan;
- expected sampler name;
- expected scheduler name;
- expected NFE;
- expected denoise;
- actual model video/audio shifts;
- VDN Runtime Report;
- checkpoint `turbo_num_steps` when declared;
- scenario profile (`auto`, `max_speed`, `reference`, ...);
- actual projection precision, so a BF16 fallback cannot be recorded as INT8/FP8;
- exact active adapter inventory;
- exact adapter strengths.

The Stage-DMD recipe must therefore report:

```text
adapters.active     = [default, turbo]
adapters.strengths  = {default: 1.0, turbo: 1.0}
```

Stage-B must report only `default=1.0`.

---

## 4. Validation layers

Use these levels in order.

### Level A — CPU/unit parity

```bash
pytest -q
python -m compileall -q vdn_h3 tests benchmarks
```

The suite covers, among other things:

- checkpoint and branch inventory;
- Diffusers -> native target mapping;
- Q/K/V slice mapping;
- SwiGLU ordering;
- PEFT alpha/rank scaling;
- low-rank factor preservation;
- grouped QKV LoRA bypass;
- VDN solve / Cholesky recurrence;
- frame statistics;
- bidirectional scan;
- complementary state gather;
- chunk-aligned windows and anchors;
- grouped attention vs dense oracle;
- exact Comfy SDPA dispatch without attention overrides;
- Flex/FA2/FA4 coverage/fallback contracts;
- tiled vs untiled branch parity;
- temporal halo handling;
- hybrid/stream slot validity;
- INT8/FP8 storage policies;
- robust pruned/curve AdaLN detection;
- e-grid AdaLN reinjection;
- benchmark recipe enforcement.

### Level B — synthetic CUDA probes

Core runtime:

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-core-gpu0.json
```

Use `--quick` first if desired.

Workstation/consumer precision and memory:

```bash
python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic-gpu0.json
```

Run independently on every target GPU. Do not infer RTX 4090 results from Blackwell or
vice versa.

### Level C — real checkpoint application

Verify:

- all 50 released branch blocks load;
- Stage-B `default` is complete;
- Stage-DMD `turbo` is complete when enabled;
- Q/K/V factors land on correct fused slices;
- `mlp.fc2` factors run through the MLP-level exact bypass (merged only under
  `lora_mode=merge`);
- curve AdaLN terms are not silently dropped;
- e-grid is found and the curve adapter is active when required;
- branch/LoRA/curve auxiliaries are registered in Comfy lifecycle;
- runtime report shows the expected adapter recipe.

### Level D — generation quality/performance

Only here make user-facing quality or speed conclusions.

---

## 5. Exact local attention contract

The released VDN local-window branch is validated against exact attention.

Normal exact VDN paths therefore do **not** pass the local windows through
`optimized_attention_override`; Sage/kitchen quantized attention must not silently alter
the trained window branch.

Exact does not mean “force the slowest PyTorch kernel”. When available,
`comfy.ops.scaled_dot_product_attention` is used so Comfy can select the fastest exact
backend for the platform:

```text
Flash attention -> cuDNN attention -> efficient attention -> math
```

This is particularly relevant on Windows/PyTorch combinations where raw
`F.scaled_dot_product_attention` may otherwise choose a poor backend.

`compat_reference` exists specifically for experiments that intentionally allow normal
Comfy attention overrides.

---

## 6. Attention calibration

The accelerated exact candidates are:

- grouped SDPA;
- FlexAttention;
- FA2 varlen decomposition;
- FA4/CuTe varlen decomposition.

Calibration compares candidates against the grouped exact oracle and persists the winner
under a signature containing hardware/software and packed geometry.

Do not treat a calibration result for one sequence geometry as evidence for another.
Representative validation should include short, primary and stress workloads.

---

## 7. AdaLN/pruned-base qualification

Pruned H3 checkpoints may represent AdaLN through a small shared curve basis. Detection
must not rely solely on `use_adaln_curves`: converted checkpoints can omit/misreport the
flag.

Kirei detects curve mode by either:

- the model flag, or
- structurally collapsed `blocks[0].adaln_proj.linear` input width.

When curve terms exist they are re-injected through `h3_silu_temb_grid.safetensors`.
They must **not** be silently skipped merely because merge mode is selected.

Validate:

- `adaln_t_table` exists;
- e-grid row count matches the table;
- e-grid full-width dimension matches the LoRA factor input;
- Runtime Report shows non-zero `curve_factors` when the active checkpoint requires them.

---

## 8. Reference comparisons

Use `profile=reference` as the numerical oracle:

- BF16 VDN projection;
- exact local SDPA without external quantized overrides;
- FP32 sensitive recurrence state;
- eager/autograd-safe recurrence path;
- factorized adapter path;
- compiler policy off.

For a divergence, compare the earliest possible tensor rather than the final MP4:

1. converted LoRA target/factor;
2. raw Q/K/V adapter output;
3. K/V short-conv feature;
4. beta / A / B statistics;
5. forward/reverse states;
6. complementary gathered state;
7. linear readout/gate;
8. projected linear delta;
9. local attention output;
10. complete block output.

---

## 9. Tiled/consumer qualification

Compare:

```text
tile_frames = 0
vs
tile_frames = 5
```

using a checkpoint with K/V temporal short convolution enabled.

Check:

- numerical parity;
- halo correctness;
- anchor zero rows;
- peak VRAM;
- branch time;
- number of compiled shapes.

Tiling is an exact memory scheduling change, not a lower-quality model.

---

## 10. Branch weight placement

### Resident

Preferred when the whole VDN branch fits with sufficient activation headroom.

### Hybrid

Keep small VDN tensors resident and stream the dominant projection representation.
This is normally the useful consumer compromise.

### Stream

CPU master for all branch weights with double-buffered device staging.

For streaming/hybrid collect:

- H2D bytes;
- copy/request count;
- ready waits;
- prefetches;
- pinned bytes;
- staging-buffer bytes;
- transfer timing.

Slot allocation is not validity. Per-slot `valid_keys` and block identity must remain
correct under gate-only/partial requests.

---

## 11. Precision qualification

### BF16

Use first for quality/fidelity. It isolates architecture and adapter correctness from
quantization.

### INT8/ConvRot

This is a Comfy-specific optimization path, not the OpenVDN published FP8 path. It must
beat BF16 on the real GPU and pass its own visual gate before it can be called an
optimization.

### FP8

OpenVDN's tuned stack quantizes large linears after LoRA merge while leaving narrow and
sensitive operations at higher precision. In Kirei the VDN projection FP8 path is
explicitly benchmarked rather than assumed to win.

For every quantized path record:

- projection precision actually resolved;
- storage/H2D difference;
- projection and complete sampler time;
- final output quality.

A fallback to BF16 is valid runtime behavior but an invalid result for an INT8/FP8-labelled
benchmark scenario.

---

## 12. Target GPU matrix

### RTX PRO 6000 Blackwell 96 GB

Qualify:

- VDN BF16 `auto`;
- VDN INT8/ConvRot;
- VDN FP8;
- `max_speed`;
- grouped vs calibrated exact attention (FA4/CuTe kernels do not exist on sm_120, so
  expect grouped or Flex to win);
- resident placement;
- experimental parallel scheduler separately.

Do not enable two-stream parallel execution by default solely because memory permits it;
it must beat the serial tuned path on the actual workload.

### RTX 4090 24 GB

Qualify:

- `auto`;
- `low_vram`;
- hybrid vs stream;
- tiled path;
- grouped/Flex;
- FA2 when installed;
- FP8 only if the actual-device probe succeeds and timing wins.

---

## 13. Generation benchmark tiers

The active scenario matrix lives in `benchmarks/scenarios.json`.

### Quick regression

```text
608×352 / 121 requested frames / Euler + simple + 8
```

Run Larry clean control and VDN BF16/INT8/FP8.

### Detail A/B

```text
960×544 / 121 / Euler + simple + 8
```

Use for micro-detail, faces/hands, texture, edge artifacts and quantization softness.

### Primary long-video performance

```text
608×352 / 241 / Euler + simple + 8
```

This is the main repeated optimization decision workload.

### Stress/crossover

```text
608×352 / 401 / Euler + simple + 8
```

If VDN still loses materially here, investigate the hot path rather than explaining the
result as a short-sequence crossover.

### Canonical release-quality geometry

```text
1344×768 / 345
```

Run:

- Stage-B reference at 50 NFE;
- Larry clean control at 8;
- Stage-DMD BF16 at 8;
- INT8/FP8 only after BF16 is understood.

H3 can align requested frame counts internally to its supported temporal grid. Record
both requested and actual latent/packed geometry in the Runtime Report.

---

## 14. Timing protocol

For each scenario:

1. record cold compilation/autotune separately;
2. perform at least one warm-up;
3. collect five warm measurements by default;
4. primary metric = median sampler-only seconds;
5. also retain peak allocated/reserved VRAM;
6. keep VAE decode/video encoding out of sampler timing;
7. record end-to-end time separately when useful.

Do not compare a warmed Turbo path against the first VDN compile/autotune run.

---

## 15. Failure triage

### Quality looks hard/patterned

First inspect the verified sampling plan. If it says `res_multistep`, `beta`, six steps or
partial denoise for Stage-DMD, the render is not a canonical VDN test.

### Quality is soft

Check:

- local VDN attention did not receive Sage/kitchen quantized override;
- adapters were not merged into a quantized base (`adapters.lora_mode` is `bypass` on
  INT8/FP8/NVFP4 bases) and `adaln_fp32` is true;
- adapter strengths are exactly 1.0 for the canonical Stage-DMD run;
- curve AdaLN factors were not dropped;
- projection precision is really BF16 in the fidelity test.

### Backend falls back

Inspect `attention_failures` and `block_fusion_error`. Optional acceleration failures are
latched intentionally.

### INT8 is slower than BF16

Treat it as a failed optimization for that GPU/geometry. Do not promote it merely because
the H3 base is INT8.

### Streaming corruption

Inspect slot block identity and `valid_keys`; allocated buffers are not proof that their
contents belong to the current block.

---

## 16. Audit status

The original optimization audit recommended, in order:

- compact low-rank LoRA runtime;
- shared QKV bypass;
- Comfy-managed branch weights;
- preallocated recurrence;
- fused temporal kernel;
- optimized frame-statistics prologue;
- compiled gather/epilogue;
- asynchronous streaming;
- automatic attention backend selection;
- production profiles and benchmark matrix.

Those architectural recommendations are represented in the current runtime. The newer
post-audit corrections are equally important for trustworthy results:

- benchmark-owned sampler/sigmas;
- canonical Euler/simple Stage-DMD trajectory;
- exact local attention protected from quantized overrides;
- Comfy exact-SDPA backend-priority dispatch;
- structural curve/pruned AdaLN detection;
- mandatory adapter/strength metadata in benchmark results.

---

## 17. CI note

The repository contains CPU/current-ComfyUI GitHub Actions jobs. CUDA qualification must
still run on the target GPU or an equivalent runner.

A red Actions badge is only a code result if GitHub actually assigned a runner and
executed the job steps. A no-runner infrastructure failure is not a pytest failure.
