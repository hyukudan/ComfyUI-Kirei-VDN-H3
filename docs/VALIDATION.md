# VDN-H3 validation and profiling

This document defines how to validate the runtime after code, ComfyUI, PyTorch, CUDA or
GPU changes. It deliberately separates numerical correctness from performance.

No hardware-specific number should be promoted into a README claim unless it was
measured with the exact command/workflow and environment being described.

## 1. Validation layers

Use four levels:

1. **CPU unit parity** — catches mathematical/API regressions cheaply.
2. **synthetic CUDA probes** — qualifies kernels, attention and memory movement without
   model weights.
3. **real checkpoint application** — verifies adapter mapping, Comfy lifecycle and
   quantized H3 integration.
4. **generation benchmark** — measures actual sampler time, peak VRAM and output
   behavior.

A successful lower level does not replace the levels above it.

---

## 2. CPU suite

From the custom-node repository:

```bash
pytest -q
```

The dependency-light tests cover:

- strict model/checkpoint inventory;
- upstream -> native adapter target mapping;
- fused QKV slice mapping;
- SwiGLU ordering;
- low-rank factor preservation;
- VDN delta-rule solve;
- frame statistics;
- forward/reverse recurrence;
- gather state math;
- reference autograd;
- grouped local attention vs dense-mask oracle;
- anchor modes;
- decomposed-plan coverage;
- shared cache ownership;
- tiled/untiled branch parity;
- temporal halo behavior;
- LoRA grouped-up fusion;
- hybrid BF16/FP8 inventory;
- calibration persistence/signature separation;
- runtime profile policy.

### Static compile

```bash
python -m compileall -q vdn_h3 tests
```

Run this before any GPU qualification.

---

## 3. Synthetic CUDA probes

### 3.1 Core optimized probe

```bash
python tests/probe_optimized_cuda.py \
  --device cuda:0 \
  --json vdn-core-gpu0.json
```

Use `--quick` for a first pass.

It exercises:

- temporal eager/Conv1d/compile/Triton parity;
- optimized recurrence vs reference recurrence;
- grouped/Flex/FA4 attention where available;
- streamed weight correctness.

### 3.2 Domestic/workstation probe

```bash
python tests/probe_domestic_cuda.py \
  --device cuda:0 \
  --json vdn-domestic-gpu0.json
```

It exercises:

- exact tiled branch vs untiled branch;
- temporal-convolution halo across tiles;
- hybrid projection streaming;
- H2D telemetry;
- actual-device FP8 `_scaled_mm` probe;
- FP8/BF16 projection parity and timing.

Run the same probe separately for every GPU intended for use.

For a two-GPU workstation where H3 is normally on GPU 0 and the display is on GPU 1:

```bash
python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic-gpu0.json
python tests/probe_domestic_cuda.py --device cuda:1 --json vdn-domestic-gpu1.json
```

Do not infer the 4090 result from a Blackwell result or vice versa.

---

## 4. Attention calibration

The calibration node benchmarks only exact candidates that are available in the active
environment:

- grouped SDPA;
- FlexAttention;
- FA2 varlen;
- FA4/CuTe varlen.

Grouped exact SDPA is the numerical oracle.

A candidate is eligible to win only when it passes:

```text
allclose(candidate, grouped, atol=2e-2, rtol=4e-2)
```

The winner is saved under an exact hardware/layout signature.

### Calibration matrix

Calibrate representative layouts rather than only one tiny sequence.

Suggested latent-frame sets:

- 22
- 41
- 102

Suggested spatial token grids should include the actual workflows used on the machine.
For each layout record:

- global rows before video;
- global rows after video;
- tokens per frame;
- exact frame count.

A calibration entry from one geometry should not be treated as evidence for another.

---

## 5. Real checkpoint application

After synthetic CUDA passes, apply the released VDN checkpoint to the real H3 base.

Verify at minimum:

- all declared branch blocks are present;
- default adapter applies completely;
- turbo adapter applies completely when enabled;
- Q/K/V factorized targets resolve to native fused QKV slices;
- native fused `mlp.fc2` factors use merge rather than activation bypass;
- pruned AdaLN curves find their e-grid;
- additional branch/LoRA/curve models appear in Comfy's model lifecycle;
- no VDN attention patch collision is silently overwritten.

### Quantized base matrix

Qualify separately:

- BF16 base, if available;
- H3 INT8/ConvRot base;
- any other quantized base intended for production.

Do not assume LoRA/weight behavior on BF16 proves compatibility with a fused quantized
base.

---

## 6. Reference comparisons

Use `profile=reference` for the exact oracle:

- exact PyTorch SDPA;
- no Comfy attention override;
- BF16 projection;
- FP32 sensitive recurrence state;
- factorized LoRA bypass;
- compile disabled;
- inference-only recurrence shortcuts disabled.

Use `compat_reference` only when comparing how ComfyUI attention overrides affect the
same VDN model.

### What to compare

For focused engineering probes, compare as early as possible:

1. Q/K/V adapter outputs;
2. frame A/B statistics;
3. prefix/suffix states;
4. gathered linear state;
5. branch readout;
6. projected linear delta;
7. local attention output;
8. complete block output.

A final video mismatch is much harder to diagnose than the first divergent tensor.

---

## 7. Tiled branch qualification

For every representative frame size, compare:

```text
tile_frames = 0
vs
tile_frames = 5
```

The test must include K/V temporal short convolution so the halo path is exercised.

Check:

- max/mean absolute error;
- first/last anchor rows remain zero in linear delta;
- peak VRAM;
- branch runtime;
- number of compiled shapes.

The tiled path is intended to be numerically equivalent within normal BF16 execution
error, not an approximate low-memory model.

---

## 8. Streaming qualification

For `stream` and `hybrid`, collect the runtime report after sampling.

Record:

- `h2d_bytes`;
- `copies`;
- `prefetch_blocks`;
- `served_stream_requests`;
- `ready_wait_events`;
- `pinned_cpu_bytes`;
- `gpu_stream_buffer_bytes`;
- `weights.transfer` stage timing.

### Expected hybrid behavior

For a normal BF16 released branch:

- the small branch tensors are resident;
- only the large `to_out_linear.weight` is streamed;
- there are two reusable GPU staging slots;
- later denoising steps should benefit from the cyclic prefetch schedule.

For FP8 interior blocks:

- streamed weight dtype is FP8;
- scale remains FP32/resident;
- edge blocks stream BF16 projections.

Any block receiving another block's projection is a correctness failure, regardless of
final output plausibility.

---

## 9. FP8 qualification

FP8 is a different numerical model path and must be evaluated separately.

First require the synthetic probe to report:

```text
available = true
```

Then measure:

- storage ratio;
- H2D bytes;
- projection time;
- complete sampler time;
- latent/video quality against BF16 baseline.

### Edge-block matrix

Test at least:

- `fp8_skip_end_blocks=4` (default);
- `0` (all transformed blocks FP8);
- optionally `2` and `6` when exploring quality/speed.

The default 4+4 BF16 policy is quality oriented.

### Important interpretation

FP8 can perturb the denoising trajectory. Pixel/latent equality with BF16 is not the
success criterion after the local projection parity has been qualified. Evaluate final
quality and consistency rather than demanding the same stochastic trajectory.

---

## 10. Target workstation matrix

For a workstation with RTX PRO 6000 96 GB + RTX 4090 24 GB, use both cards as separate
single-GPU targets.

### RTX PRO 6000

Test:

- `auto`;
- `max_speed`;
- explicit grouped;
- calibrated attention;
- FA4 if installed;
- experimental FP8;
- diagnostics off and on separately.

Primary measurements:

- resident branch viability;
- first-run compile time;
- warm sampler time;
- peak allocated/reserved VRAM;
- FA4 vs grouped crossover;
- FP8 projection and end-to-end benefit.

### RTX 4090

Test:

- `auto`;
- `low_vram`;
- explicit `stream` as a lower-VRAM fallback;
- grouped;
- Flex;
- FA2 when installed;
- experimental FP8 only if `_scaled_mm` probe passes.

Primary measurements:

- hybrid vs stream H2D waits;
- tiled vs untiled peak VRAM;
- FA2 gather overhead;
- whether the base H3 + resident small VDN tensors leave sufficient activation headroom.

---

## 11. Generation benchmark protocol

Use a fixed workflow, seed, prompt and model files when comparing runtime changes.

Measure two phases separately:

### Cold

Includes:

- first kernel compilation;
- first model/auxiliary loads;
- first mask/plan creation.

### Warm

Run after all expected compilation/cache initialization.

Report:

- sampler-only wall time;
- time per denoising step;
- end-to-end wall time;
- peak allocated VRAM;
- peak reserved VRAM;
- host RAM;
- pinned host RAM;
- H2D bytes;
- selected attention backend.

Do not combine VAE decode/video encoding with sampler timing unless the metric is
explicitly labelled end-to-end.

---

## 12. Suggested geometry matrix

At minimum:

| Frames | Purpose |
| ---: | --- |
| 22 | short smoke / regression |
| 41 | common medium workload |
| 102 | long-video pressure test |

For each frame count include at least one low/medium resolution and the largest normal
production resolution for the machine.

The important variable for attention is packed sequence geometry, not only output pixel
resolution.

---

## 13. Failure triage

### Backend falls back to grouped

Inspect `attention_failures` in the runtime report. A latched failure is intentional;
the backend is not retried on every block.

### Tiled differs from untiled

Check temporal halo, tile-local frame numbering and gather start/stop indices before
looking at the final video.

### Streaming corruption

Inspect per-slot valid-key logic and ensure a partial request did not preserve validity
from a previous block. The allocated buffer dictionary is not a validity map.

### FP8 unavailable

The actual-device `_scaled_mm` probe failed. Treat the BF16 fallback as correct behavior;
do not force an unsupported kernel based only on advertised GPU FP8 capability.

### High compile latency

Use `compile_policy=off` or `shared` for interactive use. `reduce_overhead` and
`max_autotune` are intended for repeated static geometries where warm-run speed matters.

---

## 14. Current CI note

The repository contains a GitHub Actions CPU/current-ComfyUI workflow. Hardware CUDA
qualification cannot be delegated to a standard GitHub-hosted runner and must be run on
the target machine or an equivalent GPU runner.

A CI badge/failure should only be interpreted as a code result when the workflow jobs
actually received a runner and executed their steps.
