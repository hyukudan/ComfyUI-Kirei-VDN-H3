# Performance

This page holds two things: the model that says *when* VDN can pay off, and the table of
what was actually measured. Nothing in the README or the node claims a speed-up that is
not in the measured table below.

---

## 1. Where the time goes

Per DiT block, MiniMax-H3 does four wide GEMMs (fused QKV, out, fc1, fc2) and one
attention. Dense H3 attends over the whole packed sequence; VDN-H3 attends over a local
window (the query's 5-frame chunk plus one chunk on each side, the two anchor frames and
every global row) and adds the linear branch: per-frame statistics, one `(I + A)^-1`
per frame and head, two scans, the readout and the `to_out_linear` projection.

FLOP model per block (56 heads × 128, hidden 5376, FFN 14336, text ≈ 512 tokens, audio
40 latents/s, 17 pixel frames per 5 latent frames, 2×2 patches on the 1/16 latent):

| Geometry | Latent frames | Tokens/frame | Sequence | Block GEMMs | Local softmax | Linear branch | VDN total | Dense total | Dense ÷ VDN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×352 · 121 f | 37 | 209 | 8.6 k | 6.67 (77 %) | 1.32 (15 %) | 0.67 (8 %) | 8.65 TF | 8.81 TF | **1.02×** |
| 960×544 · 121 f | 37 | 510 | 19.8 k | 15.3 | 6.3 | 1.6 | 23.2 TF | 26.5 TF | 1.14× |
| 608×352 · 241 f | 72 | 209 | 16.4 k | 12.6 (75 %) | 2.9 (17 %) | 1.3 (8 %) | 16.8 TF | 20.3 TF | 1.21× |
| 1280×736 · 145 f | 43 | 920 | 40.6 k | 31.3 (55 %) | 22.2 (39 %) | 3.4 (6 %) | 56.8 TF | 78.4 TF | 1.38× |
| 608×352 · 401 f | 119 | 209 | 26.7 k | 20.6 | 5.6 | 2.2 | 28.3 TF | 41.1 TF | 1.45× |
| 1344×768 · 345 f | 102 | 1008 | 104.5 k | 80.5 (52 %) | 66.4 (43 %) | 8.9 (6 %) | 155.8 TF | 393.5 TF | **2.53×** |

OpenVDN measured 2.65× per layer at the last geometry on a B200, which is what the
model predicts. Three things follow:

1. **Short, low-resolution clips cannot get faster.** At 608×352 with 121 frames the
   linear branch and the extra projection cost as much as the dense attention they
   replace, and VDN adds roughly a hundred kernel launches per block. There the goal is
   to break even; the win starts around 240 frames or 720p.
2. **Where VDN wins, the time is in the block GEMMs and the local softmax.** The linear
   branch itself is 6-8 %, almost all of it `to_out_linear`. Precision of the GEMMs and
   the local-attention kernel matter far more than the recurrence.
3. **A single workstation GPU is bandwidth-poorer than a B200** (about 1.8 TB/s against
   8 TB/s), so activation copies that the datacenter numbers hide are visible here.

---

## 2. Measured results

Every row must come from `benchmarks/record_result.py`, which refuses a measurement
whose sampler, scheduler, NFE, denoise, shifts, profile, resolved precision or adapters
differ from the scenario. Sampler-only seconds, median of five warm runs, peak VRAM from
`Kirei Benchmark End`.

| Date | GPU | torch / CUDA | ComfyUI | Base checkpoint | Scenario | Profile | Projection | Adapters | Attention | s / run (median) | s / NFE | Peak VRAM | Quality |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| 2026-09-04 | RTX PRO 6000 Blackwell 96 GB | 2.12.0 / 13.0 | 0.34.0 | H3 pruned INT8/ConvRot | `larry8_608x352_121` | — | — | Larry v4 @1.0 | native dense | 9.061 s | 1.133 s | 39.43 GiB | pending |
| 2026-09-04 | RTX PRO 6000 Blackwell 96 GB | 2.12.0 / 13.0 | 0.34.0 | H3 pruned INT8/ConvRot | `vdn_dmd_bf16_8step_608x352_121` | auto | bf16 | default 1 + turbo 1, bypass | Flex | 12.215 s | 1.527 s | 41.07 GiB | pending |
| 2026-09-04 | RTX PRO 6000 Blackwell 96 GB | 2.12.0 / 13.0 | 0.34.0 | H3 pruned INT8/ConvRot | `larry8_608x352_241` | — | — | Larry v4 @1.0 | native dense | 23.089 s | 2.886 s | 40.53 GiB | pending |
| 2026-09-04 | RTX PRO 6000 Blackwell 96 GB | 2.12.0 / 13.0 | 0.34.0 | H3 pruned INT8/ConvRot | `vdn_dmd_bf16_8step_608x352_241` | auto | bf16 | default 1 + turbo 1, bypass | Flex | 27.348 s | 3.419 s | 42.79 GiB | pending |
| — | RTX PRO 6000 Blackwell 96 GB | — | — | — | `vdn_dmd_max_speed_8step_608x352_241` | max_speed | as resolved | merged | — | pending | pending | pending | pending |
| — | RTX PRO 6000 Blackwell 96 GB | — | — | — | `vdn_dmd_bf16_8step_1344x768_345` | auto | bf16 | default 1 + turbo 1 | — | pending | pending | pending | pending |

Fill the table from `benchmarks/results.jsonl` (`python benchmarks/compare_results.py
benchmarks/results.jsonl`). Delete a placeholder row only when its measured row exists.

The 2026-09-04 121-frame visual pair used the same prompt and seed. A 20-frame contact
sheet showed continuous motion, stable subject/clothing, the requested cart vault and
bicycle, and no obvious hard or repeated pattern artifacts; both 5.17-second MP4 files
also contained non-silent AAC audio. Quality remains `pending` until full-motion and
audio playback is reviewed, so these rows do not authorize a same-quality speed claim.

### Release-scale stability validation (not an official benchmark row)

After fixing the Flex K/V address overflow described in [Architecture](ARCHITECTURE.md),
the canonical `vdn_dmd_bf16_8step_1344x768_345` workflow completed at 1344×768, 345
frames and 8 NFE on 2026-09-04. `auto` selected FA2 from an exact-shape calibration
(FA2 211.49 ms, grouped 219.48 ms, Flex 241.69 ms; both alternatives matched the
grouped reference within the configured tolerance). Sampling took about 221.7 s and
the whole ComfyUI prompt 281.15 s. Observed process VRAM was about 79,286 MiB and no
new `nvlddmkm` event was recorded. The same visual workload with Larry v4 completed in
about 490 s sampling / 536.23 s total. These are single end-to-end validation runs,
not five-run warm medians, so the official canonical table row remains `pending`.

### Historical points (context only, not comparable)

Recorded before the benchmark protocol existed; sampler and prompt were not captured and
the VDN runs used 4 steps on an 8-step checkpoint.

| Scenario | 608×352 · 121 f | Notes |
|---|---:|---|
| Base H3, no LoRA, forced to 4 steps | 9.37 s | diagnostic |
| Turbo LoRA, 4 steps | 10.02 s | its intended recipe |
| VDN `auto` BF16, 4 steps | 11.32 s | understepped; before the fc2/AdaLN fixes |
| VDN `auto` INT8 projection, 4 steps | 11.90 s | INT8 projection slower than BF16 here |
| Native H3, res_multistep, 20 steps | 24.54 s | quality reference |

### External reference points

| Source | Hardware | Workload | Result |
|---|---|---|---|
| OpenVDN | 1× B200, FP8, FA4 | 1344×768 · 345 f, 8 NFE | 51 s (6.41 s/NFE); dense H3 2.23 min |
| OpenVDN | 1× H200, FP8, Flex | 1344×768 · 345 f, 8 NFE | 90.5 s (11.2 s/NFE); dense H3 4.4 min |
| Saganaki22/ComfyUI-VDN-H3 v1.0 | RTX 5090 32 GB, INT8/ConvRot base, streamed branch, bypass, Sage2 patch | 1280×736 · 145 f, 8 steps | ≈ 17 s/it, ≈ 2:15 sampling |

The RTX PRO 6000 has about 10 % more SMs than a 5090 and three times the VRAM; the
Saganaki number is the first one to beat with `profile=auto`.

---

## 3. What `auto` will do on your card, and what is left to measure

| Card | Branch | Adapters | VDN projection | Attention candidates |
|---|---|---|---|---|
| RTX PRO 6000 / RTX 50xx (sm_120), BF16 base | resident | merged (exact) | BF16 | grouped, Flex-Triton, FA4 decomposition with the sm_120 mma.sync kernel when flash-attn-4 is installed; the calibration decides |
| RTX PRO 6000 / RTX 50xx, INT8 / FP8 / NVFP4 base | resident | bypass (exact, fc2 included) | BF16 | grouped, Flex-Triton, FA4 (sm_120 kernel) |
| H100 / H200 (sm_90) | resident | as above | BF16 | grouped, Flex, FA4 decomposition (wgmma) if flash-attn-4 is installed |
| B200 (sm_100) | resident | as above | BF16 | FA4 decomposition first |
| RTX 4090 24 GB | hybrid + 5-frame tiles | bypass | follows an INT8/FP8 base | grouped, Flex |

On native Windows flash-attn-4 cannot be installed (no `nvidia-cutlass-dsl` wheel), so
the sm_120 candidates there are grouped, Flex through `triton-windows` and FA2.

Explicit experiments that need a measured win *and* a passed quality gate before they
move into `auto`: INT8/ConvRot or FP8 projection (`projection_precision`), merged
adapters on quantized bases (`max_speed`), CUDA graphs (`compile_policy =
reduce_overhead`), the two-stream branch (`branch_execution = parallel`), SageAttention
on the local windows (`attention_backend = compat`).

---

## 4. Protocol

1. Confirm the environment: torch with CUDA 13 or newer (comfy-kitchen's CUDA backend
   needs it for the INT8 fast path), Triton, and the base checkpoint you actually use.
2. With the Python that runs ComfyUI, from any directory:
   `python <ComfyUI>/custom_nodes/ComfyUI-Kirei-VDN-H3/tests/probe_optimized_cuda.py --device cuda:0 --json probe-core.json`,
   the same for `probe_domestic_cuda.py`, and `probe_flex_cuda.py` once. Keep the JSON
   files next to the results.
3. Run the tiers in `benchmarks/scenarios.json` through the benchmark nodes (see
   [`benchmarks/README.md`](../benchmarks/README.md)): quick 608×352·121, detail
   960×544·121, primary 608×352·241, then stress and the canonical 1344×768·345.
4. Record cold separately; warm once; five warm runs; median sampler seconds; peak VRAM.
5. `python benchmarks/record_result.py measurement.json --seed <seed> --prompt-hash <hash> --quality-status pending`
   and update the table above.
