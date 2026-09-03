# VDN-H3 benchmark protocol

The benchmark suite compares **execution paths that target the same generation objective**.
A result is not considered a valid speed comparison merely because resolution and frame
count match.

## Comparability rule

Two results may be used for a speedup/slowdown claim only when all of these match:

- `comparison_group`;
- width and height;
- output frame count;
- denoising steps / NFE;
- seed;
- scheduler and relevant schedule/shifts;
- prompt/conditioning payload;
- model objective (for example the 4-step distilled/Turbo objective).

A 4-step base-model diagnostic run and a normal 20-step native run are **not** in the
same comparison group. The former measures backbone overhead at an intentionally
under-stepped setting; the latter is the normal native-quality baseline.

`compare_results.py` enforces the structural part of this rule and refuses to rank rows
inside a comparison group when resolution, frames, steps, seed or scheduler differ.

## Primary benchmark geometries

### Short regression: 608x352, 121 frames

Keep this because it is fast enough for frequent iteration and already has historical
measurements.

Primary comparable group:

```text
distilled4_608x352_121
```

Compare Turbo conventional, VDN BF16, VDN INT8/ConvRot and other 4-step VDN profiles
inside that group.

### Primary long-video benchmark: 608x352, 241 frames

This is the main performance target. It is long enough for the local-window/linear
attention trade to matter much more than at 121 frames while still being practical for
repeated workstation testing.

Primary comparable group:

```text
distilled4_608x352_241
```

At minimum run:

- Turbo conventional, 4 steps;
- VDN `auto`, BF16 projection, 4 steps;
- VDN `auto`, INT8/ConvRot projection, 4 steps;
- VDN `max_speed`, 4 steps.

### Stress/crossover benchmark: 608x352, 401 frames

This is the long-sequence stress test and is close to the regime in which the original
VDN work reports its strongest scaling advantages. It is not required after every small
change, but it is required before claiming that a performance optimization improves
long-video scaling.

Comparable group:

```text
distilled4_608x352_401
```

At minimum compare Turbo conventional and the best qualified VDN path.

## Native-quality baseline

A normal native MiniMax-H3 generation at 20 steps belongs to its own comparison group,
for example:

```text
native20_608x352_121
```

It may be used to answer a product-level question such as "how long does normal native
generation take compared with a 4-step accelerated workflow?" It must **not** be used to
claim a kernel or VDN speedup relative to a 4-step distilled run.

## Measurement protocol

For each scenario:

1. restart or otherwise establish the same clean model state when comparing first-run
   behavior;
2. record the first execution separately as `cold`;
3. perform at least one warm-up execution;
4. collect **five measured warm executions** by default;
5. report the **median sampler time** as the primary number;
6. also record end-to-end time and peak VRAM;
7. save the Kirei Runtime Report for every VDN scenario.

Do not average first-run compilation/autotune into steady-state sampler time.

Recommended result fields:

```json
{
  "scenario_id": "vdn_auto_int8_608x352_241",
  "comparison_group": "distilled4_608x352_241",
  "width": 608,
  "height": 352,
  "frames": 241,
  "steps": 4,
  "seed": 1234,
  "scheduler": "fixed-name-and-settings",
  "prompt_hash": "...",
  "run_kind": "warm",
  "sampler_seconds": 12.34,
  "end_to_end_seconds": 14.56,
  "peak_vram_bytes": 123456789,
  "runtime_report": {}
}
```

The results file is JSONL: one execution per line.

Summarize it with:

```bash
python benchmarks/compare_results.py benchmarks/results.jsonl
```

The script ranks only scenarios sharing one comparison group and one benchmark
invariant. Different objectives are deliberately kept separate.

## Historical 121-frame measurements

These measurements use 608x352 and 121 output frames:

| Configuration | Steps | Time | Interpretation |
| --- | ---: | ---: | --- |
| Base without LoRA | 4 | 9.37 s | diagnostic only; not normal native quality |
| Turbo conventional | 4 | 10.02 s | principal 4-step baseline |
| VDN `auto`, BF16 | 4 | 11.32 s | directly comparable with Turbo 4-step |
| VDN `auto`, INT8 | 4 | 11.90 s | directly comparable with Turbo/VDN 4-step |
| Native standard | 20 | 24.54 s | native-quality baseline, separate objective |

These numbers predate later runtime changes and must not be presented as current results.
Their value is methodological: they establish why step count and generation objective are
part of benchmark identity.

## What constitutes an optimization

For a change to be called a speed optimization on a given geometry:

- it must beat the previous qualified path in the **same comparison group**;
- numerical/quality qualification for that path must still pass;
- the reported gain must use warm median sampler time;
- a gain caused solely by fewer steps, lower resolution, fewer frames or a different
  generation objective is not an implementation speedup.

For the current 4-step target, Turbo conventional is the control path to beat. VDN's
main performance question is therefore whether its long-sequence scaling overtakes that
control as the frame count grows from 121 -> 241 -> 401.
