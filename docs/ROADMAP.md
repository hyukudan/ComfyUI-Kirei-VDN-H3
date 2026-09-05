# Roadmap

What is done, what the first measurements say, and what is left, in the order it should
be done. Every pending item names the measurement that closes it. Dates and numbers come
from `benchmarks/results.jsonl`, `benchmarks/artifacts/` and `docs/PERFORMANCE.md`.

---

## 1. Done (0.3.0, unreleased)

| Area | State |
|---|---|
| Math parity with OpenVDN | window semantics, recurrence, gates, adapters, AdaLN fp32 activation on full AdaLN bases, `mlp.fc2` exact bypass |
| `auto` policy | merge on BF16 bases, exact bypass on INT8/FP8/NVFP4/MXFP8, BF16 projection when resident, VRAM budget against total minus base model |
| Attention dispatch | calibrated grouped / Flex / FA2 / FA4 per geometry, calibration store v3 keyed by node, torch, CUDA, driver, Triton, flash-attn versions and installed backends; peak-memory guard prefers Flex; dispatch reason on every call |
| Flex | one dynamic compiled call, recompile floor 64, reachable-offset guard against the int32 K/V addressing fault seen at 1344×768 · 345 f |
| sm_120 | never treated as B200; flash-attn-4's mma.sync kernel is a calibration candidate where it can be installed (not native Windows) |
| Block fusion | fused pre/post kernels, compiled SwiGLU on non-INT8 bases |
| Tooling | benchmark nodes with recipe enforcement, recorder, Runtime Report (adapters, precision, attention, layout rows, environment), CUDA probes runnable from any directory with real qkv strides |
| Docs | README, ARCHITECTURE, PERFORMANCE, VALIDATION, benchmarks README, CHANGELOG |
| Tests | CPU suite green; `probe_flex_cuda.py` twelve-length recompile regression green on the RTX PRO 6000 |

## 2. What the first measurements say (RTX PRO 6000, 2026-09-04)

| Scenario | Larry v4 control | VDN `auto` (BF16 projection, bypass, Flex) | Ratio | FLOP model |
|---|---:|---:|---:|---:|
| 608×352 · 121 f, 8 NFE | 9.061 s | 12.215 s | 0.74× | 1.02× |
| 608×352 · 241 f, 8 NFE | 23.089 s | 27.348 s | 0.84× | 1.21× |
| 1344×768 · 345 f, 8 NFE (single run, not a table row) | ≈ 490 s | ≈ 221.7 s | ≈ 2.2× | 2.53× |

Two conclusions drive the order below.

1. **At the canonical geometry VDN already delivers what the model predicts.** The
   official row needs five warm runs and a quality review, nothing else.
2. **At short geometries there is a fixed overhead the FLOP model does not see.** At
   241 frames VDN spends 4.26 s more than the control over 8 steps: about 0.53 s per
   step, about 10.6 ms per block, while the branch FLOPs for that geometry are worth a
   few milliseconds per step. That gap is latency and memory traffic, not arithmetic:
   sequential per-frame scan kernels, the low-rank bypass GEMMs, the extra projection,
   block-mask and launch overhead. It is the main optimisation target.

All quality columns are `pending`: no same-quality speed claim exists yet.

---

## 3. Pending, in order

### P1. Close the short-geometry gap (608×352 · 241 f first)

1. Run `vdn_dmd_bf16_8step_608x352_241` with `diagnostics = true` and read the per-stage
   CUDA-event timings in the Runtime Report: window softmax, frame statistics, solve,
   scans, readout, `to_out_linear`, adapter terms, mask creation. Record the split in
   `docs/PERFORMANCE.md`.
2. Attack the largest stage, each as an explicit experiment with a benchmark row:
   - scans: CUDA graphs (`compile_policy = reduce_overhead`) or a chunked scan that
     replaces the per-frame `baddbmm` loop;
   - overlap: `branch_execution = parallel` (second stream) once the branch is graphed;
   - adapters: `vdn_dmd_max_speed_8step_608x352_241` quantifies the bypass GEMMs against
     merged adapters on the INT8 base; the quality gate decides whether any of it enters
     `auto`;
   - projection: INT8 / FP8 `to_out_linear` measured again on this stack (the earlier
     INT8 result was slower than BF16).
3. Success criterion: VDN at 241 frames within 5 % of the control or faster, with the
   Stage-DMD quality gate passed. Until then the README keeps saying that short clips
   do not get faster.

### P2. Official canonical rows and the remaining tiers

- `vdn_dmd_bf16_8step_1344x768_345` and `larry8_1344x768_345`: five warm runs, cold run
  recorded separately, peak VRAM, `record_result.py`.
- `960x544_121`, `608x352_401` tiers for the crossover curve.
- Keep `attention_calibration` artifacts next to each date (FA2 won at 104k tokens,
  Flex at 8k and 16k; this is the evidence for the calibration policy).

### P3. Quality review, then speed claims

- Same prompt, same seed: Stage-B / 50 (`reference`) against Stage-DMD / 8 (`auto`) and
  against the Larry control; full-motion playback and audio, not only contact sheets.
- A/B pairs that justify the `auto` policy with images, not only theory: `adaln_fp32`
  on/off, bypass vs merged adapters on the INT8 base, BF16 vs INT8 projection.
- Set `quality_status` on every row; only then quote a speed-up in the README.

### P4. Flex follow-ups

- Restore the tensor-captured mask geometry after a GPU check. With Python ints in the
  closure, `create_block_mask` recompiles once per distinct layout (bounded by the
  recompile floor of 64) and past the floor falls back to an eager S × S mask.
- Avoid the contiguous copies the int32 guard makes at the canonical geometry by
  producing contiguous V (and Q/K where needed) once upstream of the window call.
- Re-run `probe_flex_cuda.py` after each change; it must still print `PASS`.

### P5. Exact-attention candidates (measured win and quality gate required)

- Copy-free grouped attention (per-group LSE merge, no K/V gathers): FA2 beat grouped
  by 4 % at 104k tokens, so a copy-free grouped path could take the lead.
- FA4 on sm_120: WSL2 or a Linux ComfyUI; the v3 signature re-calibrates by itself once
  `flash-attn-4` imports. Never force it with `--no-deps` on native Windows.
- SageAttention on the local windows (`attention_backend = compat`): approximate, needs
  the quality gate.

### P6. Layouts with many global rows (REF2VA, I2V references, keyframes)

- Add a benchmark scenario with reference video rows so the peak-memory guard and the
  Flex choice are exercised by the recorder, not only by unit tests.
- Report the video / global row split for such a run and confirm no OOM at 24 GB with
  `low_vram`.

### P7. Reference parity on GPU

- Run OpenVDN's reference window and branch code on the same tensors as Kirei's
  optimized path (Level A of `docs/VALIDATION.md`) and record max/mean error per stage.
  This is the test that turns "parity by construction" into "parity by measurement".

### P8. Infrastructure and release

- CI: the GitHub-hosted runner is not picking up jobs; fix the workflow, and add a
  manual, self-hosted job that runs the three CUDA probes and uploads their JSON.
- Registry: set `PublisherId` in `pyproject.toml` before `comfy node publish`.
- Tag 0.3.0 when P2 and P3 have at least the canonical row with a reviewed quality
  status; move the CHANGELOG entries from "unreleased".

---

## 4. Not planned

- Multi-GPU / Ulysses sharding: the runtime represents one compute GPU.
- Quantized projections or merged adapters in `auto` without a measured win and a
  passed quality gate.
- Speed claims from single runs or from geometries other than the recorded scenarios.
