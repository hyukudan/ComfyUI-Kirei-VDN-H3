# Changelog

## 0.3.0 (unreleased)

Quality-first runtime for single-GPU workstations, with the fixes found by the
September 2026 audit against OpenVDN and current ComfyUI.

### Fidelity

- **Exact `mlp.fc2` adapter terms.** The turbo adapter targets `ff.net.2` in every
  block; those factors were always merged into the weight (a requantization on
  INT8/FP8 bases) even in bypass mode. They now run as an exact activation-space term
  on the parent MLP, keeping the native fused fc2 kernel.
- **AdaLN activated from the fp32 time embedding**, as OpenVDN's patched diffusers does
  (ComfyUI rounds the embedding to BF16 first). New `adaln_fp32` switch, on by default.
- `auto` adapters: merged on BF16 bases (exact, no runtime GEMMs), exact bypass on
  INT8/FP8/NVFP4/MXFP8 bases. `max_speed` still merges everywhere, quality-gated.
- `auto` keeps the VDN `to_out_linear` projection in BF16 when the branch is resident;
  INT8/FP8 projections are explicit experiments.

### Runtime

- Consumer/workstation Blackwell (sm_120: RTX 50xx, RTX PRO 6000) is no longer treated
  as datacenter Blackwell: the FA4/CuTe decomposition is only tried on sm_90/sm_100.
- Attention dispatch estimates the K/V copies grouped attention would make for the
  layout and picks Flex before autotuning when reference/keyframe rows make them too
  large (REF2VA/I2V layouts).
- The fused block path compiles the SwiGLU whenever ComfyUI will not fuse it (every base
  that is not INT8).
- `detect_base_precision` recognises NVFP4, MXFP8, W4A4 and W4A8 comfy-kitchen layouts.
- `auto` branch placement budgets against total VRAM minus the base model size instead
  of the free VRAM seen before ComfyUI loads the base.

### Nodes and reports

- Every input of **Kirei Apply VDN-H3** has a tooltip; the profile list leads with
  `auto`, `max_speed`, `low_vram`, `reference`.
- Runtime Report: `adaln_fp32`, GPU family, attention `dispatch_reason`, and a
  video / text / other-global row breakdown of the packed layout.
- Benchmark recorder requires the active adapter recipe and `lora_mode`.

### Documentation

- README rewritten around install, the three settings that matter, the sampling recipe
  (Euler / simple / 8 / cfg 1) and what `auto` decides per GPU.
- New `docs/PERFORMANCE.md` with the FLOP model and the measured-results table.
- Architecture and validation docs synchronised with the runtime.

## 0.2.0

- Native ComfyUI MiniMax-H3 integration: exact VDN window semantics, bidirectional
  recurrent branch, Triton temporal kernel, shared compile caches, resident/hybrid/stream
  branch storage, INT8/FP8 projection, calibrated exact attention dispatch, factorized
  LoRA bypass, curve AdaLN reinjection, recipe-aware benchmark nodes and runtime report.
