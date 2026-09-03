# VDN-H3 benchmark protocol

The benchmark suite compares **execution paths that target the same generation objective**.
Matching resolution and frame count is not enough: step recipe, conditioning, schedule and
output quality target are part of benchmark identity.

## Checkpoint recipe: 8 denoising steps

The currently qualified VDN checkpoint declares:

```text
turbo_num_steps = 8
```

Therefore the main Turbo/VDN comparison uses **8 steps**. Earlier 4-step measurements
were under-stepped relative to the checkpoint recipe and are retained only as historical
diagnostics.

This matters for both performance and image quality. A 4-step VDN or Turbo run may be
faster, but it is not the intended recipe for this checkpoint and can lose fine detail.
It must not be used as the production control for the 8-step model.

The benchmark protocol also does **not** assume that VDN quality equals Turbo quality just
because both run 8 steps. The available VDN/DMD checkpoint is earlier in training than
the Turbo adapter, so quality must be qualified explicitly before a speed result is
promoted as a same-quality speedup.

---

## Comparability rule

Two results may be used for a speedup/slowdown claim only when all of these match:

- `comparison_group`;
- `quality_target`;
- `recipe`;
- width and height;
- output frame count;
- denoising steps / NFE;
- seed;
- scheduler and schedule/shifts;
- prompt/conditioning payload;
- model objective.

For the current VDN/Turbo checkpoint, the comparable distilled recipe is:

```text
steps = 8
recipe = turbo_num_steps_8_same_prompt_seed_scheduler
```

A 4-step base-model diagnostic, an under-stepped 4-step Turbo/VDN run and a normal
20-step native run are all separate objectives.

`compare_results.py` enforces the structural part of this rule. It refuses to rank rows
inside one comparison group if any required benchmark invariant differs.

---

# Quality gate

A benchmark can be technically comparable but still fail the same-quality requirement.
Each result therefore carries:

```text
quality_status = pending | qualified | failed | diagnostic
```

Use the statuses as follows:

- `pending`: timing is valid, but no same-quality speed claim may be made yet;
- `qualified`: output passed the agreed A/B quality check and is eligible for a speed claim;
- `failed`: output is visibly below the control target; timing may be useful for engineering,
  but it is not a same-quality optimization;
- `diagnostic`: deliberately non-production configuration such as the historical 4-step
  under-stepped runs.

For visual A/B qualification, keep prompt, seed, scheduler and all conditioning fixed and
inspect at least:

- fine texture/detail and local sharpness;
- subject/identity consistency;
- motion quality and temporal continuity;
- small-object/facial/hand stability where relevant;
- edge ringing, smearing, flicker and repeated-pattern artifacts;
- overall contrast and micro-detail, not only global composition.

A speed optimization is not considered product-qualified until the target VDN profile
passes this gate against the control path.

---

# Benchmark tiers

## 1. Short regression: 608×352, 121 frames, 8 steps

Use for frequent iteration and regression checks.

Comparable group:

```text
distilled8_608x352_121
```

Run at minimum:

- Turbo conventional, 8 steps;
- VDN `auto`, BF16 projection, 8 steps;
- VDN `auto`, INT8/ConvRot projection, 8 steps.

This is no longer the final performance verdict; it is the quick feedback loop.

## 2. Primary visual-detail A/B: 960×544, 121 frames, 8 steps

This is the main quality-oriented A/B because the higher spatial load exposes softness,
loss of texture and small-detail degradation much more clearly than 608×352.

Comparable group:

```text
quality8_960x544_121
```

Recommended order:

1. Turbo conventional 8-step control;
2. VDN `auto` BF16 8-step;
3. only after BF16 quality is understood, VDN INT8/ConvRot 8-step.

Do not interpret INT8-vs-BF16 timing before checking whether quantization changes the
visual result enough to matter.

If 960×544 remains too soft to distinguish the quality ceiling clearly, escalate the
quality A/B to approximately **1360×768 / 121 frames / 8 steps** while keeping the same
prompt, seed and scheduler. That is a conditional quality stress test, not the routine
performance benchmark.

## 3. Primary long-video performance benchmark: 608×352, 241 frames, 8 steps

This is the main end-to-end performance target. It is long enough for the local-window /
linear-memory trade to become meaningful while remaining practical for repeated runs.

Comparable group:

```text
distilled8_608x352_241
```

Run:

- Turbo conventional 8-step control;
- VDN `auto` BF16;
- VDN `auto` INT8/ConvRot;
- VDN `max_speed` after the quality path is qualified.

This group should be used for normal optimization decisions.

## 4. Stress/crossover benchmark: 608×352, 401 frames, 8 steps

Comparable group:

```text
distilled8_608x352_401
```

This is required before claiming improved long-video scaling. It answers the important
question: if VDN is close to or behind Turbo at 121 frames, does its asymptotic advantage
actually appear at 241/401 frames?

At minimum compare:

- Turbo conventional 8-step;
- VDN `auto` BF16;
- VDN `auto` INT8/ConvRot if it already passed the quality gate.

If VDN still loses materially at 401 frames, treat that as an implementation/hot-path
problem rather than explaining it away as a short-sequence crossover.

---

# Native-quality baseline

A normal native MiniMax-H3 generation at 20 steps remains useful, but it belongs to its
own comparison group:

```text
native20_608x352_121
```

It answers a product-level question such as "how long does normal native generation take
versus the accelerated distilled workflow?" It must not be used as the control for a
VDN kernel/runtime speedup.

Likewise, a base H3 run forced to 4 steps is only a backbone-overhead diagnostic and is
not normal native quality.

---

# Measurement protocol

For each scenario:

1. use the exact scenario recipe from `benchmarks/scenarios.json`;
2. keep prompt, seed, conditioning, scheduler and shifts fixed across one comparison group;
3. record the first execution separately as `cold`;
4. run at least one warm-up after compilation/autotune;
5. collect **five warm measured executions** by default;
6. report **median sampler time** as the primary performance number;
7. also record end-to-end time and peak VRAM;
8. save the Kirei Runtime Report for every VDN scenario;
9. record `quality_status` independently of timing;
10. never mix the cold compile/autotune run into steady-state timing.

## In-graph sampler timing

Use the two benchmark nodes so Turbo, VDN and native are timed over exactly the same
segment:

```text
model
  -> Kirei Benchmark Start
       -> MODEL -> sampler -> LATENT -> Kirei Benchmark End.after
       -> benchmark_token ------------> Kirei Benchmark End.benchmark_token
```

Set `scenario_id` on **Kirei Benchmark Start** to an id from `scenarios.json`.

The Start node synchronizes the model CUDA device before opening the interval and can
reset peak-memory statistics. The End node synchronizes before closing it and outputs:

- `sampler_seconds`;
- peak allocated/reserved VRAM;
- run kind (`cold` or `warm`);
- optional embedded VDN Runtime Report when the patched model is connected.

Connect the sampler LATENT directly to `End.after`. If you connect a decoded video or a
post-VAE node instead, the measurement intentionally includes that extra work and is no
longer the standard sampler benchmark.

Both benchmark nodes invalidate ComfyUI's execution cache on every prompt so repeated
warm measurements really execute.

## Recording a measurement

Save the JSON emitted by **Kirei Benchmark End**, then merge it with the scenario metadata:

```bash
python benchmarks/record_result.py measurement.json \
  --seed 1234 \
  --scheduler "fixed-name-and-settings" \
  --prompt-hash "<hash>" \
  --quality-status pending
```

`record_result.py` appends a normalized row to `benchmarks/results.jsonl` by default.
For VDN measurements it also inspects `runtime_report.checkpoint_recipe.turbo_num_steps`.
If the checkpoint reports 8 but the selected scenario requests another step count, the
result is rejected rather than recorded.

Recommended normalized result row:

```json
{
  "scenario_id": "vdn_auto_int8_8step_608x352_241",
  "comparison_group": "distilled8_608x352_241",
  "quality_target": "checkpoint_declared_distilled8",
  "recipe": "turbo_num_steps_8_same_prompt_seed_scheduler",
  "quality_status": "pending",
  "quality_gate_required": true,
  "comparable": true,
  "width": 608,
  "height": 352,
  "frames": 241,
  "steps": 8,
  "seed": 1234,
  "scheduler": "fixed-name-and-settings",
  "prompt_hash": "...",
  "checkpoint_turbo_num_steps": 8,
  "run_kind": "warm",
  "sampler_seconds": 12.34,
  "peak_vram_bytes": 123456789,
  "runtime_report": {}
}
```

The results file is JSONL: one execution per line.

Summarize it with:

```bash
python benchmarks/compare_results.py benchmarks/results.jsonl
```

The comparator still shows a technical ranking when the structure matches, but
`speed_claim_eligible` becomes `true` only after every path that requires a quality gate
is marked `qualified`.

---

# Historical 4-step measurements

These measurements used 608×352 and 121 output frames, but **not** the checkpoint's
8-step recipe:

| Configuration | Steps | Time | Interpretation |
| --- | ---: | ---: | --- |
| Base without LoRA | 4 | 9.37 s | diagnostic only; not normal native quality |
| Turbo conventional | 4 | 10.02 s | historical under-stepped Turbo result |
| VDN `auto`, BF16 | 4 | 11.32 s | historical under-stepped VDN result |
| VDN `auto`, INT8 | 4 | 11.90 s | historical under-stepped VDN result |
| Native standard | 20 | 24.54 s | normal native-quality baseline, different objective |

These numbers predate later runtime changes and are no longer the active VDN/Turbo
benchmark. They remain useful only to explain earlier observations and to verify that a
future result is not accidentally being compared against the wrong recipe.

---

# What constitutes an optimization

A change can be called a same-quality speed optimization only when:

- it beats the previous qualified path in the **same comparison group**;
- it uses the checkpoint-declared 8-step recipe for the current distilled VDN/Turbo model;
- its `quality_status` is `qualified` against the control target;
- the gain is measured from warm median sampler time;
- peak VRAM remains acceptable for the intended hardware profile;
- no gain comes solely from fewer steps, lower resolution, fewer frames or changed
  conditioning.

The current engineering sequence is therefore:

```text
608×352 / 121f / 8-step  -> quick regression
960×544 / 121f / 8-step  -> detail/quality gate
608×352 / 241f / 8-step  -> primary performance decision
608×352 / 401f / 8-step  -> long-video scaling/crossover proof
```
