# Follow-up plan — October 10, 2026

Based on Claude CLI's read-only review of Kirei 0.3.1, VDN-H3-Plus and VELA,
followed by source checks. Review hypotheses are not reproduced failures.
Current evidence: 155 local tests, successful GitHub CPU/current-ComfyUI jobs,
one 768p VDN video and Larry controls at strengths 1.0 and 0.65. This does not
qualify every runtime policy or establish end-to-end OpenVDN parity.

## Priority 0: correctness and qualification

| Work | Reason | Acceptance |
|---|---|---|
| Reproduce bypass coexistence with ComfyUI's real bypass LoRA hook in both injection orders | Kirei replaces `module.forward`; another replacement can prevent safe eject | Correct additive output and clean inject/eject; preserve clone A/B/A and stale-eject tests. Consider post-hooks only after reproduction |
| Calibrate with real, distinct Q/K/V | Current automatic calibration reuses Q as all three inputs, weakening its parity check | A deliberately wrong anchor mask fails the parity gate; valid backends pass |
| Add pinned OpenVDN source oracles | Existing tests compare against Kirei's own reference implementations | Window/anchor, branch, hybrid and timestep tests against a recorded upstream SHA |
| Qualify default `auto` and a consumer GPU | Current video forced BF16 projection and compile off on a 96 GB card | Actual default compile/fusion run and 24 GB hybrid/stream run complete; record runtime and visual checks |

## Priority 1: compatibility and exact memory improvements

- The compiled block forward lacks native `attention=None`. Reproduce a
  `blocks_replace` call and preserve the native fallback when an override is supplied.
- Review FA2/FA4 K/V copies and gather lifetimes. Remove a copy only after CUDA
  parity and five-run peak-memory/timing checks; grouped savings do not prove FA2 savings.
- Report the selected AdaLN source (full grid or affine companion), with identity
  verification status. Existing full grids are not bound to the loaded base table.
- Review fc2 bypass activation lifetime; require unchanged output and measured peak
  reduction before reordering work around the native fused operation.
- Make short full-cover attention honor the documented exact policy, or explicitly
  retain and document Comfy attention overrides.
- Add a pinned-ComfyUI integration job alongside the rolling compatibility job.

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
