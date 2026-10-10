# Follow-up plan — October 10, 2026

Based on Claude CLI's read-only review of Kirei 0.3.1, VDN-H3-Plus and VELA,
followed by source checks. Review hypotheses are not reproduced failures.
Sol verified the source hypotheses and implemented the confirmed fixes in 0.3.2.
The full clean-Core suite passes 193 tests, including 20 reduced pinned OpenVDN
source oracles. This does not qualify every runtime policy or establish end-to-end
checkpoint/video parity.

## Priority 0: correctness and qualification

| Work | Status in 0.3.2 | Acceptance evidence |
|---|---|---|
| Native bypass coexistence in both injection orders | Fixed after reproducing both failures; independent linear post-hooks | Real Comfy hook two-cycle tests, shared A/B/A, stale eject and rollback |
| Calibrate with real, distinct Q/K/V | Implemented; three timing runs, cache schema 4 | Deliberately wrong anchor mask passes old QQQ check and fails distinct-input check |
| Pinned OpenVDN source oracles | Implemented; source SHA and clean checkout checked | Twenty reduced FP32 window/anchor, branch, hybrid and paired timestep tests |
| Default `auto` and consumer GPU qualification | Separate generation validation | Require completed runs and runtime/visual evidence; CPU tests do not qualify these routes |

## Priority 1: compatibility and exact memory improvements

- Implemented native `attention=None` compatibility; supplied overrides fall back
  to the original block. A regression executes the `blocks_replace` argument form.
- Removed redundant FA2/FA4 K/V copies and shortened gather lifetimes. FA2 on the
  recorded synthetic 768p layout: five warm medians 140.257→139.379 ms, peak
  518 MiB lower, bit-identical output. FA4 hardware validation remains pending.
- Implemented `curve_source` reporting: type, path, file SHA256 and table identity
  status. Legacy unbound grids explicitly report unverified.
- Review fc2 bypass activation lifetime; require unchanged output and measured peak
  reduction before reordering work around the native fused operation.
- Implemented exact short full-cover dispatch outside explicit `compat`.
- Added pinned-ComfyUI/OpenVDN CI alongside rolling compatibility.

## Priority 2: measured optimization

- Evaluate deferred `cholesky_ex` error handling without weakening numerical checks.
- Quality-gate the hybrid/stream quantized projection decision; do not describe it
  as exact merely because attention and adapter placement are exact.
- Fill the performance table with warm five-run medians and reviewed video/audio,
  including Stage-B at 50 steps. Cold VDN and warm Larry times are not comparable.
- Test Larry strength separately from VDN's released adapter recipe. The 0.65
  control looked less hardened in sampled frames, but its composition changed.

VELA's selective Sage, anchor pruning and approximate QKV are excluded from the
default policy until separately qualified. Exact three-window batching already lost
on the measured RTX 4090 geometries. Retained global scratch pools and mixed-grid
interfaces are not proposed for this update.
