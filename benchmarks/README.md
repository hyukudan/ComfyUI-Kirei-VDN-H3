# VDN-H3 benchmark protocol

This benchmark suite has two different jobs and keeps them deliberately separate:

1. **product / recipe-faithful comparison** — compare systems that target the same output
   objective while allowing each system to run the recipe it was actually trained for;
2. **technical / same-NFE comparison** — isolate runtime implementation changes inside
   VDN by holding NFE and the VDN recipe fixed.

This distinction is essential for MiniMax-H3 Turbo vs VDN-H3.

## The released recipes are different

The conventional community/Comfy Turbo LoRA is a **4-step** recipe.

The released OpenVDN `stage-dmd-step-250` checkpoint is **not** that same 4-step LoRA
simply attached to VDN. OpenVDN initializes its `turbo` adapter from the external Turbo
LoRA and then DMD2-trains it inside the VDN manifold as a **50 -> 8 NFE** student.

The released Stage-DMD generator is therefore:

```text
H3 base
+ VDN linear branch
+ adapters/default
+ adapters/turbo
+ 8 NFE
+ video shift 12.0
+ audio shift 3.0
```

The Stage-B fidelity/reference recipe is:

```text
H3 base
+ VDN linear branch
+ adapters/default
+ turbo OFF
+ 50 NFE
+ video shift 12.0
+ audio shift 3.0
```

The benchmark must never force conventional Turbo to 8 steps merely to match VDN, and
must never run Stage-DMD VDN at 4 steps and call that its production recipe.

`benchmarks/scenarios.json` is the source of truth for these recipes.

---

# Two comparison classes

## A. Product comparison: same objective, intended recipe

A product group asks:

> For the same prompt, seed, resolution, frame count and quality objective, how fast is
> each system when each one uses the recipe it was actually trained for?

Typical group:

```text
Turbo conventional        4 NFE
VDN Stage-DMD BF16         8 NFE
VDN Stage-DMD INT8         8 NFE, only after quality qualification
VDN Stage-DMD max_speed    8 NFE, only after quality qualification
```

Different NFE counts are **allowed** here. They are part of the model recipe, not a
benchmarking cheat.

A product comparison requires the following to match:

- output width/height;
- output frame count;
- prompt/conditioning payload;
- seed;
- H3 scheduler family;
- video shift 12.0;
- audio shift 3.0;
- declared quality target.

Each path may have a different `recipe_id` and NFE count.

The comparator emits these under:

```text
product_comparisons
```

A product speed claim is enabled only when all paths requiring a visual quality gate are
marked `qualified`.

## B. Technical comparison: same NFE and same VDN recipe

A technical group asks a different question:

> Is this implementation change actually faster when the model, schedule and amount of
> denoising work are unchanged?

Examples:

```text
VDN Stage-DMD BF16
vs VDN Stage-DMD INT8/ConvRot
vs VDN Stage-DMD max_speed
```

All entries in a technical group must match:

- resolution;
- frames;
- NFE;
- seed;
- exact scheduler name/settings;
- scheduler family;
- prompt hash;
- VDN recipe id;
- video/audio shifts;
- quality target.

The comparator emits these under:

```text
technical_comparisons
```

This is the correct place to answer questions such as “does INT8 make VDN faster than
BF16?”

---

# Canonical recipes

The scenario file currently defines:

| Recipe | NFE | Video shift | Audio shift | Intended use |
| --- | ---: | ---: | ---: | --- |
| `turbo_conventional_4` | 4 | 12.0 | 3.0 | Conventional MiniMax-H3 Turbo control |
| `vdn_stage_dmd_8` | 8 | 12.0 | 3.0 | Released VDN Stage-DMD (`default + turbo`) |
| `vdn_stage_b_50` | 50 | 12.0 | 3.0 | VDN port-fidelity / teacher-quality diagnostic |
| `native_standard_20` | 20 | 12.0 | 3.0 | Existing local native-quality reference |

The native 20-step baseline is useful as a product reference but is not a same-NFE
control for a distilled model.

---

# Benchmark tiers

## 1. Quick regression — 608×352, 121 frames

Product group:

```text
fewstep_product_608x352_121
```

Run:

- conventional Turbo at **4 NFE**;
- VDN Stage-DMD BF16 at **8 NFE**;
- VDN Stage-DMD INT8 at **8 NFE** after BF16 quality is understood.

The VDN BF16/INT8 runs also belong to a strict technical group:

```text
vdn_dmd8_608x352_121
```

This is the fast feedback loop, not the final quality verdict.

## 2. Higher-detail A/B — 960×544, 121 frames

Product group:

```text
fewstep_product_960x544_121
```

This is useful for seeing softness, micro-detail loss, edge artifacts and quantization
changes more clearly than 608×352.

Again:

```text
Turbo = 4 NFE
VDN-DMD = 8 NFE
```

Do not force either model onto the other's step count.

## 3. Primary long-video benchmark — 608×352, 241 frames

Product group:

```text
fewstep_product_608x352_241
```

This is the **primary repeated performance benchmark** because it is long enough for the
VDN local-window/linear-memory trade to matter without making every iteration extremely
expensive.

Run:

- Turbo conventional, 4 NFE;
- VDN Stage-DMD `auto` BF16, 8 NFE;
- VDN Stage-DMD `auto` INT8, 8 NFE after quality gate;
- VDN Stage-DMD `max_speed`, 8 NFE after its own quality gate.

The VDN variants also share:

```text
vdn_dmd8_608x352_241
```

for same-NFE engineering comparisons.

## 4. Stress/crossover — 608×352, 401 frames

Product group:

```text
fewstep_product_608x352_401
```

This is required before making claims about long-sequence scaling. If VDN remains slower
at 401 frames, treat that as an implementation/hot-path problem rather than explaining
it away as a short-sequence crossover.

## 5. Canonical OpenVDN quality/fidelity geometry — 1344×768, 345 frames

OpenVDN trains/evaluates around latent `48×84`, which corresponds to the 768p-class
`1344×768` geometry, and its released few-step model uses 345 output frames / 8 NFE.

### Stage-B port-fidelity gate

Run:

```text
vdn_stage_b_bf16_50step_1344x768_345
```

with:

```text
Stage-B checkpoint
default adapter = 1.0
turbo adapter = OFF
projection = BF16
50 NFE
video shift = 12.0
audio shift = 3.0
```

This test answers whether the VDN architecture + Stage-B adapter port itself preserves
quality before Turbo is involved.

### Stage-DMD canonical quality gate

Product group:

```text
fewstep_product_release_1344x768_345
```

Run:

- conventional Turbo: 4 NFE;
- VDN Stage-DMD BF16: 8 NFE;
- VDN Stage-DMD INT8: 8 NFE only after BF16 passes.

For the VDN release path the required adapter recipe is:

```text
default = 1.0
turbo   = 1.0
```

This is the most important quality benchmark in the suite.

---

# Quality gate

Timing alone cannot turn a numerically different path into an optimization.

Each result carries:

```text
quality_status = pending | qualified | failed | diagnostic
```

Use:

- `pending` — timing valid, output not yet approved;
- `qualified` — output meets the target for that product group;
- `failed` — output visibly below the target; timing remains engineering information;
- `diagnostic` — intentionally non-production or historical setup.

For A/B qualification inspect at least:

- subject/identity consistency;
- texture and micro-detail;
- temporal coherence and motion;
- facial/hand/small-object stability where applicable;
- flicker, repeated patterns, ringing and smearing;
- prompt following and composition;
- audio/video behavior when audio is part of the workflow.

Keep prompt, seed and conditioning identical within the product group.

---

# Measurement protocol

For each active scenario:

1. select the scenario directly in **Kirei Benchmark Start**;
2. use the recipe from `scenarios.json` exactly;
3. connect Start MODEL directly to the sampler;
4. connect sampler LATENT directly to **Kirei Benchmark End.after**;
5. for VDN, connect the same patched MODEL to Benchmark End so Runtime Report is embedded;
6. record the cold run separately;
7. perform at least one warm-up after compilation/autotune;
8. collect five warm measured executions by default;
9. use median `sampler_seconds` as the primary number;
10. also record peak VRAM and, when useful, end-to-end time.

The Start node now records the model's actual H3 sampling metadata:

```text
sampling.class
sampling.video_shift
sampling.audio_shift
```

`record_result.py` rejects a result if the measured shifts do not match the scenario
recipe.

For VDN Stage-DMD it also requires a VDN Runtime Report and checks the checkpoint's
`turbo_num_steps=8` declaration. If a newer Runtime Report exposes named adapters and
strengths, the recorder additionally verifies the expected adapter inventory.

---

# Recording and summarizing

Save the JSON emitted by **Kirei Benchmark End** and normalize it with:

```bash
python benchmarks/record_result.py measurement.json \
  --seed 1234 \
  --scheduler "euler/minimax-h3" \
  --prompt-hash "<hash>" \
  --quality-status pending
```

Then summarize:

```bash
python benchmarks/compare_results.py benchmarks/results.jsonl
```

The output contains two independent sections:

```text
product_comparisons
technical_comparisons
```

A product group can legitimately contain 4-step Turbo and 8-step VDN because that is the
trained recipe comparison. A technical VDN group cannot mix 4 and 8 NFE.

---

# Historical measurements

The earlier 608×352 / 121-frame measurements remain stored for context:

| Configuration | NFE | Time | Correct interpretation |
| --- | ---: | ---: | --- |
| Base without LoRA | 4 | 9.37 s | diagnostic only; native base was under-stepped |
| Turbo conventional | 4 | 10.02 s | Turbo was using its intended 4-step recipe |
| VDN `auto`, BF16 | 4 | 11.32 s | VDN Stage-DMD was incorrectly run at 4 instead of 8 |
| VDN `auto`, INT8 | 4 | 11.90 s | same under-stepped VDN issue, plus older runtime |
| Native standard | 20 | 24.54 s | local native-quality baseline, different recipe |

These are not mixed into current speed claims because prompt/seed/scheduler metadata was
not fully recorded and the VDN runs were outside the released Stage-DMD recipe.

---

# What counts as an optimization

For a **technical VDN optimization**:

- same `technical_group`;
- same 8-NFE Stage-DMD recipe;
- same prompt/seed/scheduler/shifts;
- quality gate passes when the change is numerically different;
- warm median sampler time improves.

For a **product-level acceleration claim**:

- same `product_group` and output objective;
- each model uses its own intended recipe;
- Turbo remains 4 NFE;
- VDN-DMD remains 8 NFE;
- every required output is quality-qualified;
- warm median sampler time is compared end to end over the same sampler segment.

The main sequence is therefore:

```text
608×352 / 121f  -> quick product + technical regression
960×544 / 121f  -> detail/quality A/B
608×352 / 241f  -> primary long-video performance decision
608×352 / 401f  -> long-sequence scaling proof
1344×768 / 345f -> canonical Stage-B / Stage-DMD quality-fidelity gate
```
