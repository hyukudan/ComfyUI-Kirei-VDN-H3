# VDN-H3 benchmark protocol

The benchmark suite is designed to prevent a workflow from silently changing the model's
denoising trajectory. Resolution, frame count and NFE are not enough: **sampler,
scheduler, denoise, H3 shifts, adapters and runtime profile are part of benchmark
identity**.

The current source of truth is `benchmarks/scenarios.json`.

---

# Canonical sampling recipes

## OpenVDN Stage-DMD

The released Stage-DMD checkpoint is benchmarked as:

```text
sampler       = euler
scheduler     = simple
steps / NFE   = 8
denoise       = 1.0
video shift   = 12.0
audio shift   = 3.0
adapters      = default 1.0 + turbo 1.0
```

This corresponds to OpenVDN's MiniMax-H3 flow/Euler inference trajectory.

**Do not use `res_multistep` for this checkpoint.** `res_multistep` is a base-model
sampler; attaching the Stage-DMD/Turbo adapters while keeping that sampler changes the
trajectory and can produce hard, patterned or otherwise misleading output.

## OpenVDN Stage-B fidelity reference

```text
sampler       = euler
scheduler     = simple
steps / NFE   = 50
denoise       = 1.0
video shift   = 12.0
audio shift   = 3.0
adapters      = default 1.0
turbo         = OFF
projection    = BF16
```

This is the port-fidelity test. If Stage-B/50 already looks wrong, investigate the VDN
architecture/default adapter mapping before blaming Stage-DMD Turbo.

## Larry MiniMax-H3 Turbo v4 quality control

Larry v4 supports 4–8 steps, with its README noting that **6–8 steps improve motion and
detail over the 4-step minimum**. For the clean A/B control against VDN we therefore use:

```text
sampler       = euler
scheduler     = simple
steps         = 8
denoise       = 1.0
LoRA strength = 1.0
video shift   = 12.0
audio shift   = 3.0
```

On current ComfyUI, stock Euler is valid because MiniMax-H3's audio/video dual schedule
is handled natively. Larry's dedicated Turbo Sampler is also valid for its own workflow,
but the benchmark uses stock Euler so Larry and VDN share the exact sampler trajectory.

## Native/base reference

The base-model reference is separate:

```text
sampler       = res_multistep
scheduler     = simple
steps         = 20
denoise       = 1.0
```

It must not be copied into a Turbo/VDN benchmark.

## Local visual workflow reference

`local_visual_ref_608x352_121` records the locally preferred workflow for context:

```text
hybrid b30-49
SageAttention
Larry v4 strength 0.6
Euler
beta scheduler
6 steps
```

This is **not** the canonical Larry/OpenVDN benchmark. It exists to distinguish “what
looked good locally” from “what reproduces the released inference recipe”.

---

# The benchmark owns SAMPLER and SIGMAS

The most important rule is that benchmark runs must not reuse sampler widgets from an
existing workflow.

Use this graph:

```text
Kirei Benchmark Scenario
        │ scenario_id
        ▼
Kirei Benchmark Sampling ◄──── MODEL
        │
        ├── SAMPLER ───────────────┐
        ├── SIGMAS ────────────────┤
        └── recipe_token           │
                 │                 │
                 ▼                 │
Kirei Benchmark Start ◄──── MODEL │
        │ MODEL                    │
        └──────────────────────────┼──► SamplerCustomAdvanced
                                  │
SamplerCustomAdvanced ◄───────────┘
        │ LATENT
        ▼
Kirei Benchmark End
```

`Kirei Benchmark Sampling` uses the same current Comfy primitives as
`KSamplerSelect + BasicScheduler`:

```text
comfy.samplers.sampler_object(sampler_name)
comfy.samplers.calculate_sigmas(model_sampling, scheduler_name, steps)
```

For Stage-DMD that means the node itself constructs **Euler + simple + 8 NFE**.

`Kirei Benchmark Start` requires the opaque `recipe_token` produced by that node. A
hand-configured sampler cannot generate a valid token, so an inherited `res_multistep`
widget cannot accidentally be recorded as a canonical VDN result.

---

# Automatic validation

`record_result.py` rejects the measurement if any of these differ from the selected
scenario:

- sampler name;
- scheduler name;
- number of NFE;
- denoise;
- video shift;
- audio shift;
- VDN checkpoint `turbo_num_steps` declaration;
- VDN profile (`auto`, `max_speed`, `reference`, ...);
- requested projection precision when the Runtime Report exposes it;
- named adapter inventory/strengths when exposed by the Runtime Report.

In particular, these are invalid Stage-DMD measurements:

```text
res_multistep + simple + 8
Euler + beta + 8
Euler + simple + 6
Euler + simple + 8 + denoise 0.8
```

The canonical Stage-DMD tuple is exactly:

```text
Euler / simple / 8 / denoise 1.0 / shifts 12,3
```

---

# Benchmark tiers

## 1. Quick regression — 608×352, 121 requested frames

H3 may internally align a requested frame count to its supported temporal grid. That is
normal and is not itself a benchmark error.

Product group:

```text
quality8_608x352_121
```

Run:

- Larry v4 clean control — Euler/simple/8, strength 1.0;
- VDN Stage-DMD BF16 — Euler/simple/8;
- VDN Stage-DMD INT8 — Euler/simple/8, after BF16 quality is understood;
- VDN Stage-DMD FP8 — Euler/simple/8, after BF16 quality is understood.

This is the fast regression loop, not the final long-video verdict.

## 2. Higher-detail A/B — 960×544, 121 frames

Product group:

```text
quality8_960x544_121
```

Use it to inspect micro-detail, edge artifacts, skin/hair texture, faces, hands and
quantization-induced softness or patterning.

## 3. Primary long-video benchmark — 608×352, 241 frames

Product group:

```text
quality8_608x352_241
```

This is the **primary repeated performance benchmark**. It is long enough that VDN's
local-window/linear-memory trade should be meaningful while still being practical for
multiple warm runs.

Run at minimum:

- Larry v4 8-step clean control;
- VDN BF16;
- VDN INT8;
- VDN FP8;
- VDN `max_speed` after its quality gate.

## 4. Stress/crossover — 608×352, 401 frames

Product group:

```text
quality8_608x352_401
```

If VDN still loses materially here, treat it as an implementation/hot-path problem rather
than attributing it to short-sequence crossover.

## 5. Canonical OpenVDN quality geometry — 1344×768, 345 frames

Use:

```text
vdn_stage_b_bf16_50step_1344x768_345
larry8_1344x768_345
vdn_dmd_bf16_8step_1344x768_345
vdn_dmd_int8_8step_1344x768_345
vdn_dmd_fp8_8step_1344x768_345
```

This tier answers two separate questions:

1. does the Stage-B VDN port preserve the model at 50 NFE?;
2. does Stage-DMD at 8 NFE preserve acceptable quality relative to the clean Larry
   control at the same Euler/simple/8 trajectory?

---

# Product vs technical comparisons

`compare_results.py` produces two sections.

## `product_comparisons`

Compares different model recipes that target the same quality objective **only when the
actual denoising trajectory matches**. The current canonical Larry-vs-VDN groups are all:

```text
8 NFE
Euler
simple
denoise 1.0
shifts 12/3
same geometry
same prompt hash
same seed
```

## `technical_comparisons`

Compares only entries from the **same `recipe_id`**. This is where BF16, INT8, FP8 and
`max_speed` VDN are ranked against each other.

Larry can share the product group with VDN but never contaminates the VDN technical
ranking.

---

# Quality gate

Each result carries:

```text
quality_status = pending | qualified | failed | diagnostic
```

A speed claim is not eligible until every required path in the product comparison is
`qualified`.

Check at least:

- subject/identity consistency;
- prompt following;
- texture and micro-detail;
- facial/hand/small-object stability;
- temporal coherence and motion;
- flicker, ringing, repeated textures or hard/patterned artifacts;
- audio/video behavior where audio is part of the graph.

Use the same prompt, conditioning, seed and requested geometry within a product group.

---

# Measurement protocol

For each scenario:

1. select it in **Kirei Benchmark Scenario**;
2. wire its geometry values into the latent/video workflow;
3. connect its `scenario_id` to **Kirei Benchmark Sampling**;
4. connect the sampled MODEL to **Kirei Benchmark Sampling**;
5. connect the resulting `SAMPLER` and `SIGMAS` to `SamplerCustomAdvanced`;
6. connect its `recipe_token` to **Kirei Benchmark Start**;
7. feed Benchmark Start's MODEL output to that same sampler;
8. connect the sampler LATENT directly to **Kirei Benchmark End.after**;
9. for VDN, also connect the patched MODEL to Benchmark End;
10. record cold compile/autotune separately;
11. warm once;
12. collect five warm runs by default;
13. report median `sampler_seconds` plus peak VRAM.

Never include VAE decode or MP4 encoding in the standard sampler interval.

---

# Recording results

Save the JSON emitted by Benchmark End and run:

```bash
python benchmarks/record_result.py measurement.json \
  --seed 1234 \
  --prompt-hash "<hash>" \
  --quality-status pending
```

`--scheduler` is optional and is now only a human-readable label. The real sampler and
scheduler are taken from the verified `sampling_plan` embedded in the measurement.

Summarize with:

```bash
python benchmarks/compare_results.py benchmarks/results.jsonl
```

---

# Historical results

Older measurements remain useful as historical context but are **not canonical** if the
full sampling plan was not recorded.

The previously observed VDN output generated with `res_multistep + simple` must not be
used to judge Stage-DMD quality. Repeat it through `Kirei Benchmark Sampling` first.

Likewise, the local hybrid/Sage/Larry-0.6/Euler-beta-6 workflow can be used as a visual
reference, but it is deliberately outside the clean Larry/OpenVDN benchmark.

---

# What counts as an optimization

A technical VDN change is an optimization only when:

- same geometry;
- same prompt/seed;
- same `vdn_stage_dmd_8` recipe;
- **Euler + simple + 8 NFE + denoise 1.0**;
- same H3 shifts 12/3;
- output passes the relevant quality gate;
- warm median sampler time improves;
- peak VRAM remains acceptable for the target GPU.

The routine sequence is:

```text
608×352 / 121f  -> quick regression
960×544 / 121f  -> visual-detail gate
608×352 / 241f  -> primary performance benchmark
608×352 / 401f  -> long-sequence scaling proof
1344×768 / 345f -> Stage-B and Stage-DMD canonical quality gate
```
