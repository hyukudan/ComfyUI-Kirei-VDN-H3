# Bypass clone ownership regression — 2026-09-05

Historical audit. The ownership fix was integrated into `main` and the running
ComfyUI installation on October 10, 2026. The current full suite passes 155 tests;
the counts below describe the original September audit.

The prior code guarded individual hook instances, but distinct ModelPatcher clones share
one inner model and create different hook objects. Injecting a second clone without ejecting
the first could wrap the prior adapter. A merge/zero-bypass clone installed no callback at all.

The fix coordinates the active hook set on the shared model. Clone takeover ejects the old
set in reverse order before injecting the new set. Stale ejects do not remove the new owner.
Partial failures roll back. An unrelated replacement of module.forward is not overwritten:
the operation raises rather than silently destroying another extension's hook.
Merge/zero-factor clones install an empty ownership callback to displace active bypass hooks.
Adapter arithmetic and merge/bypass defaults remain unchanged.

Regression tests cover repeated injection, A/B/A transitions, stale eject, pre-existing and
later unrelated hooks, rollback, and bypass-to-merge transitions. Full CPU suite: **145 passed**;
focused ownership suite: **6 passed**. Ruff and git diff --check passed.

This is evidence of a lifecycle defect, not proof that the ninja's visible defects came from it.
Live native ModelPatcher load/unload, cancellation and GPU-quality comparisons remain to run
before claiming deployment or image-quality improvement. No copied T8 GPL code was introduced.
