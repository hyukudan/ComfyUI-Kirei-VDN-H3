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
  as datacenter Blackwell: the FA4/CuTe decomposition is only assumed to win on
  sm_90/sm_100 (wgmma / tcgen05 kernels). On sm_120 flash-attn-4's mma.sync kernel stays
  a calibration candidate; the Runtime Report names the generation as `fa4_kernel`.
- Attention dispatch separates peak-memory risk from aggregate K/V-copy bandwidth.
  Peak-risk layouts select Flex without attempting an unsafe calibration; bandwidth-only
  warnings still calibrate the installed exact backends and persist the measured winner.
- Flex now guards the real reachable storage offset of fused Q/K/V views before calling
  Inductor. A 1344×768, 345-frame layout exceeded int32 only through its physical stride;
  the generated K/V address arithmetic wrapped and caused repeatable driver MMU faults.
  Dangerous views are made contiguous before Flex, while smaller zero-copy layouts stay
  unchanged.
- The fused block path compiles the SwiGLU whenever ComfyUI will not fuse it (every base
  that is not INT8).
- `detect_base_precision` recognises NVFP4, MXFP8, W4A4 and W4A8 comfy-kitchen layouts.
- `auto` branch placement budgets against total VRAM minus the base model size instead
  of the free VRAM seen before ComfyUI loads the base.
- Flex uses one dynamic compiled call and raises dynamo's recompile limit to 64. Static
  specialisation let Inductor select unsafe int32 K/V offset arithmetic for large fused
  H3 views; dynamic compilation plus the storage-offset guard keeps those addresses safe
  and avoids eager Flex (a dense S x S score matrix) across changing packed lengths.
- Calibration store v3: signatures include the node version, torch/CUDA/cuDNN/driver,
  Triton, flash-attn 2 / flash-attn-4 versions and the installed backend inventory, so a
  node update or a newly installed backend re-measures; v2 files are ignored. Automatic
  calibration averages three steady calls instead of trusting one sub-millisecond sample.
- Dispatch reason rewritten on every resolution (`explicit`, `calibrated`, `flex` guard,
  `autotuned`, `heuristic`); it no longer survives a calibration hit.

### Nodes and reports

- Every input of **Kirei Apply VDN-H3** has a tooltip; the profile list leads with
  `auto`, `max_speed`, `low_vram`, `reference`.
- Runtime Report: `adaln_fp32`, GPU family, attention `dispatch_reason`, and a
  video / text / other-global row breakdown of the packed layout.
- Benchmark recorder requires the active adapter recipe and `lora_mode`.
- Benchmark Scenario now exposes profile/projection as the same Comfy combo types used
  by Apply, so the documented wiring passes prompt validation. Checkpoint metadata also
  carries `turbo_num_steps` into the Runtime Report for strict recipe validation.
- Runtime Report: `fa4_available` next to `fa4_kernel`, `backends_available`, the dynamo
  recompile limit and the calibration environment.

### Documentation

- README rewritten around install, the three settings that matter, the sampling recipe
  (Euler / simple / 8 / cfg 1) and what `auto` decides per GPU.
- New `docs/PERFORMANCE.md` with the FLOP model and the measured-results table.
- Architecture and validation docs synchronised with the runtime.
- CUDA probes run from any directory with the ComfyUI Python; the core probe uses the
  real qkv strides, the real calibration path and times FA2; the Flex probe is a
  twelve-length recompile regression. Windows notes: flash-attn-4 is not installable
  natively, Flex needs `triton-windows`.
- Installation documents the checkpoint-local `h3_silu_temb_grid.safetensors` needed by
  curve-form bases instead of relying silently on the legacy MiniMax-H3-Turbo sibling.
- The quality checklist distinguishes the optional fp32-SiLU patch on full AdaLN bases
  from the native fp32 curve path used by pruned bases.

## 0.2.0

- Native ComfyUI MiniMax-H3 integration: exact VDN window semantics, bidirectional
  recurrent branch, Triton temporal kernel, shared compile caches, resident/hybrid/stream
  branch storage, INT8/FP8 projection, calibrated exact attention dispatch, factorized
  LoRA bypass, curve AdaLN reinjection, recipe-aware benchmark nodes and runtime report.
