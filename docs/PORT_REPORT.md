# Private feasibility report: VideoDeltaNet → ComfyUI

Date: 2026-09-03  
Status: private alpha; real checkpoint and short generation smoke validated
Target machine: RTX PRO 6000 Blackwell 96 GB + RTX 4090 24 GB, Windows

## Executive conclusion

The port is technically feasible and the RTX PRO 6000 is a strong single-GPU target.
The correct architecture is a custom ComfyUI model-patch node that replaces the 50
MiniMax H3 DiT attention modules with Comfy-native hybrid attention wrappers while
leaving the two text-refiner attention blocks dense.

We should not copy the upstream Diffusers model wholesale into ComfyUI. ComfyUI already
has a native, memory-managed H3 implementation, its VAE/audio pipeline, conditioning,
quantized loaders, and existing workflows. The port should reuse all of that and add
only VDN's local-softmax branch, linear branch, gates, checkpoint loader, and optional
kernel backends.

The published VDN-H3 weights were downloaded for the user-requested local engineering
test after the territorial restriction was surfaced. The
published VDN-H3 weights inherit the MiniMax H3 community license, whose model card
states that the applicable territory excludes the European Union. The source code is
Apache-2.0, but downloading/using the weights in Spain needs authorization from
MiniMax. This report is engineering analysis, not legal advice.

## Evidence captured locally

### Hardware

- GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB, compute
  capability 12.0.
- GPU 1: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9.
- Driver: 610.88.
- The current shell has `CUDA_VISIBLE_DEVICES=1`, so the existing Comfy environment
  sees only the 4090 unless the launcher overrides it to `0`.

### Existing ComfyUI assets

- Native MiniMax H3 implementation exists in
  `comfy/ldm/minimax/model.py` with 50 DiT blocks.
- Existing BF16 FL2VA base: 37.46 GiB.
- Existing INT8 ConvRot FL2VA base: 19.53 GiB.
- Existing Ref2VA and hybrid FL2VA/Ref2VA checkpoints are also present.
- Existing `minimax_h3_turbo_v4_step600_ema.safetensors` has 518 tensors in
  Comfy-native fused-QKV format. This is the community checkpoint from which VDN's
  DMD turbo adapter was initialized; the final VDN turbo adapter is not identical and
  must eventually be loaded from the VDN artifact.
- ComfyUI checkout commit: `95d755cd8107a72258d452b5d3657273d571f07d`.
- The checkout is dirty with unrelated user changes. The port must remain isolated and
  must not edit or reset those files during prototype work.

### Runtime

- Python 3.12.9.
- PyTorch 2.12.0+cu130.
- CUDA runtime 13.0.
- Triton installed.
- FlashAttention 2.9.1 installed.
- `flash_attn.cute` absent.
- PyTorch FlexAttention is available.

### Kernel probes on RTX PRO 6000

1. BF16 CUDA matmul: passed.
2. Compiled FlexAttention default/Triton backend: passed.
3. FlexAttention with `kernel_options={"BACKEND": "FLASH"}`: failed at compile time
   with the expected message that the CuTe FlashAttention library is unavailable.

This means a correctness-first port can run now with FlexAttention/Triton. Matching
the upstream Blackwell fast path requires a separate environment with FlashAttention
4/CuTeDSL (and likely the upstream-recommended PyTorch 2.13 + CUDA 12.9 combination),
or careful qualification of a newer compatible stack. We must not replace the active
Comfy environment until this is tested in an isolated environment.

## Upstream VDN artifact

The released 8-NFE artifact is structurally:

- `linear_branch/model.safetensors`: 4.28 GB.
- `adapters/default`: Stage-B VDN LoRA.
- `adapters/turbo`: DMD 8-step LoRA.
- Total VDN add-on: 5.46 GB.
- Base H3 files are much larger but are already available locally in Comfy format.

Released transform configuration:

- 50 DiT attention blocks transformed.
- Local window: 5-frame chunks, radius 1 (own, previous, and next chunk; normally a
  15-frame local window).
- First and last latent frames are dense anchors in both row and column directions.
- Linear attention rule: `vdn_solve`.
- Linear head dimension: 128.
- K/V short convolution enabled.
- Text initializes both directional linear states.
- Linear accumulator matrix A uses FP32.
- Softmax output gate enabled.

## Structural differences that the port must bridge

| Concern | Upstream Diffusers VDN | Native ComfyUI H3 | Required adapter |
| --- | --- | --- | --- |
| DiT list | `transformer_blocks` | `blocks` | Prefix rename |
| Q/K/V | `to_q`, `to_k`, `to_v` | fused `qkv_proj` | One fused projection, split output |
| Q/K norms | `norm_q`, `norm_k` | `q_norm`, `k_norm` | Attribute adapter |
| Output | `to_out[0]` | `out_proj` | Attribute adapter |
| Text blocks | `token_refiner.refiner_blocks` | `token_refiner.blocks` | Prefix rename |
| Feed-forward | `ff.net.0.proj`, `ff.net.2` | `mlp.fc1`, `mlp.fc2` | Prefix/name rename; preserve SwiGLU half ordering |
| Final modulation | `norm_out.linear` | `final_layer.adaln_proj.linear` | Name mapping |
| Packed layout | Separate `SequenceLayout` | `PackedLayout` in `minimax_payload` | Runtime geometry adapter |
| Attention shape | Batch-first Diffusers | flattened packed sequence | Comfy-native wrapper contract |

## Recommended runtime design

### Node contract

Create `Apply VideoDeltaNet H3`:

Inputs:

- `MODEL`: an already loaded native Comfy MiniMax H3 base.
- `vdn_checkpoint`: local folder containing `model_spec.json`, branch weights, and
  adapters.
- `backend`: `auto | flex | flash | reference`.
- `precision`: initially `bf16`; later `fp8`.
- `teacher_mode`: diagnostic only.

Output: cloned, reversibly patched `MODEL`.

The node must reject non-H3 models, wrong layer/head dimensions, truncated artifacts,
unknown transform versions, missing tensors, and incompatible base variants.

### Reversible patching

Use `ModelPatcher.clone()` and `add_object_patch()` for:

- `diffusion_model.blocks.N.attn` → `ComfyHybridAttention(original_attn, ...)`, for
  N=0..49.
- A lightweight `diffusion_model.forward` wrapper that obtains
  `minimax_payload["layout"]`, converts it once per shape, stores it in a `ContextVar`
  for the duration of the forward, and always restores the prior value in `finally`.

The context variable avoids modifying Comfy core just to thread layout through four
call levels. A future upstream-quality implementation should instead add an explicit
layout argument or a documented transformer option to Comfy core.

Object patches are applied before `ModelPatcher.load()`, so the added branch modules
participate in the model traversal and device movement. For the first prototype, force
full model residency on the 96 GB GPU and reject low-VRAM/offload mode; add proper
partial-load support only after correctness is established.

### Comfy-native hybrid attention

The wrapper should:

1. Call the existing `qkv_proj` once.
2. Retain raw Q/K/V for the linear branch.
3. Apply Comfy's existing fused RMSNorm+RoPE path to Q/K for local softmax.
4. Calculate local/anchor attention through VDN's FlexAttention mask.
5. Apply the learned softmax gate and existing `out_proj`.
6. Run VDN's bidirectional linear branch only over target-video rows.
7. Project through learned `to_out_linear` and add only to target-video output rows.

Audio, reference image/video rows, and prompt rows remain in the exact softmax path.
Only the target video span participates in the far linear branch, matching upstream.

### Layout conversion

Comfy's target video segment is identified by the segment whose kind is `video`.
From `PackedLayout.signature = (text_len, latent_t, latent_h, latent_w, audio_t)`:

- `num_frames = latent_t`.
- `frame_height = ceil(latent_h / 2)` after H3's 2×2 spatial patch.
- `frame_width = ceil(latent_w / 2)`.
- `tokens_per_frame = (video_end-video_start) / num_frames` and must equal
  `frame_height * frame_width`.
- Text is the first `text` segment, not all global/non-video rows.

Every relationship must be asserted. Silent inference from sequence length is unsafe
because references and audio alter the packed sequence.

## Checkpoint conversion strategy

Do not generate a second 40 GB merged base checkpoint. Load the existing base and the
5.46 GB VDN add-on separately.

### Branch tensors

Rename `transformer_blocks.N` to `blocks.N`. The wrapper deliberately retains VDN
submodule names (`linear_attention`, `to_out_linear`, `softmax_gate`), so the remainder
of branch keys can load without semantic conversion.

### LoRA tensors

Two choices exist:

1. Convert each Diffusers adapter to native Comfy fused-QKV LoRA tensors. This is good
   for ordinary Comfy LoRA handling but Q/K/V may use separate A matrices, so fusing
   into one low-rank pair can increase rank or require block-diagonal construction.
2. Merge adapter deltas directly into cloned model weights through ModelPatcher. This
   preserves exact math and avoids materializing a new base checkpoint.

Use option 2 first. For Q/K/V, calculate each `B @ A` in FP32 and patch the appropriate
row slice of `qkv_proj`. Map output, MLP, AdaLN, refiner, and final modulation targets
directly. Never mutate the shared base model in place; all deltas must belong to the
cloned patcher so unpatching is exact.

The original-layout SwiGLU convention differs from Diffusers. When translating an
original/Comfy `mlp.fc1` adapter into Diffusers, upstream swaps gate/value halves. In
our direction (Diffusers VDN adapter → native Comfy), perform the inverse swap on the
LoRA B rows for `ff.net.0.proj`.

## Phased implementation

### Phase 0 — completed in this workspace

- Clone and audit upstream source.
- Audit current Comfy H3 internals and installed assets.
- Identify exact architectural and naming differences.
- Probe both FLASH and Flex backends on the RTX PRO 6000.
- Add dependency-light layout/key mapping modules and unit tests.

### Phase 1 — CPU/synthetic correctness

- Vendor only required Apache-2.0 VDN modules with LICENSE/NOTICE attribution.
- Implement `ComfyHybridAttention` against tiny synthetic dimensions.
- Compare dense/full-cover wrapper output against original Comfy attention.
- Test local-window masks, anchors, text state, and target-video-only addition.
- Load a synthetic branch checkpoint strictly and test reversible object patches.

Acceptance: full-cover teacher parity within BF16 tolerance; all branch tensors load
exactly once; no mutation leaks into the unpatched base.

### Phase 2 — real weights

- Download only `stage-dmd-step-250` (5.46 GB), not the duplicate H3 base.
- Verify artifact hashes/spec and inventory every tensor without allocating the full
  checkpoint.
- Apply VDN default + turbo adapters to the existing BF16 FL2VA base.
- Run 5 frames / low spatial resolution / one denoising step.
- Increase to 41 frames, then 121, then the released 345-frame workload.

Acceptance: finite latents/audio, no missing/unexpected keys, deterministic repeat,
no global model mutation, successful VAE decode.

### Phase 3 — performance

- Establish BF16 Flex baseline.
- Add upstream inference-only fused bodies where compatible.
- Create an isolated FA4/PyTorch environment and rerun the CuTe probe.
- Add FP8 only after BF16 output comparisons pass.
- Benchmark total Comfy execution, sampler-only, VAE, peak VRAM, and compile warm-up
  separately. Never compare our end-to-end time with upstream denoising-only numbers.

### Phase 4 — packaging

- Workflow for existing H3 T2VA/FL2VA conditioning.
- Backend diagnostics node and JSON run record.
- Windows and WSL instructions.
- Tests against the pinned Comfy commit plus a current upstream Comfy version.
- Private release first; upstream/public release only after license review.

## Go/no-go assessment

Hardware: **GO**. The 96 GB Blackwell GPU is sufficient for a BF16-first prototype and
leaves room for the ~4.28 GB branch on top of the existing 37.46 GiB base. Actual peak
activations at 345 frames still need measurement.

Software architecture: **GO**. The mismatch is localized and does not require replacing
Comfy's sampler, VAE, audio pipeline, or conditioning nodes.

Fast kernel stack: **PARTIAL GO**. Flex/Triton works now; FA4/CuTe does not exist in the
active environment.

Weights/legal: **NO-GO until authorization** for use of the published weights in Spain.
Continue with source and synthetic tensors meanwhile.

## Audit of the newly published Saganaki22 port

Repository reviewed: `Saganaki22/ComfyUI-VDN-H3`, commit
`9d037fab19bcf398937743fc4371c98229b283ad` (2026-09-03). The repository appeared
after the initial survey and is a real native ComfyUI VDN-H3 port. It is a useful
reference, but it should not be installed unchanged yet.

### What is worth reusing

- Its window/anchor geometry and central VDN recurrence agree with the official
  OpenVDN formulas in the covered tests.
- It confirms that a `ModelPatcher`-based native H3 integration is practical.
- Its grouped-SDPA fallback is portable and its FlexAttention implementation ran on
  the local RTX PRO 6000 with maximum BF16 disagreement `0.00390625` versus grouped
  SDPA in a CUDA smoke test.
- It requires no new Python dependencies and contains no subprocess, telemetry,
  automatic download, pickle load, or arbitrary code execution path found by this
  audit.

### Blocking defects found

1. **Full-cover softmax-gate shape bug.** In `vdn_h3/hybrid.py`, the dense/full-cover
   path leaves attention flattened as `[S, H*D]` and multiplies it by a gate shaped
   `[S, H, 1]`. This crashes on short clips, where full coverage is especially likely.
2. **Missing token-refiner LoRAs.** The adapter key mapper does not translate the
   official token-refiner attention keys into Comfy's fused `qkv_proj`/`out_proj`.
   Bypass mode silently skips those missing destinations, so a run can complete while
   not reproducing the released checkpoint.
3. **Unmanaged caches.** Branch weights, checkpoint tensors, Flex block masks, and
   gather indices are retained in global caches outside ComfyUI memory accounting.
   The Flex cache also keys only on device type (`cuda`), not `cuda:0` versus
   `cuda:1`, which is unsafe on this dual-GPU machine.
4. **Excess QKV/LoRA memory.** The implementation retains raw full-sequence Q/K/V
   during branch readout and represents fused QKV LoRAs with a large block-diagonal
   matrix full of zeros. This increases both peak VRAM and GEMM cost.
5. **Unsafe checkpoint path resolution.** A workflow can use `..` components to make
   the loader read JSON/safetensors outside `models/vdn`; the resolved path is not
   checked for containment.
6. **Suite is not green under pytest.** Local result: 6 passed, 1 failed, 1 error.
   The individual mathematical scripts pass, but pytest has a missing fixture and
   import-order/stub contamination. Important end-to-end parity cases are absent.

### Performance findings on the target machine

- The PRO 6000 should prefer a managed resident branch cache: the extra ~4.3 GB fits
  comfortably in 96 GB. Streaming roughly 90 MB per block measured 6.33 ms from
  pageable host memory and 2.50 ms from pinned memory in a synthetic transfer probe.
- The current manual five-tap temporal convolution measured 10.86 ms at a realistic
  geometry; grouped depthwise `conv1d` measured 3.17 ms and used about 982 MiB less
  peak allocation. Its different BF16 accumulation order needs official parity tests
  before becoming the exact/default path.
- The current implementation makes about 1.40 GiB of extra raw Q/K/V copies per block
  at the audited long-video geometry. V can be view-only; Q/K lifetime can be reduced
  by scheduling the linear branch before in-place norm/RoPE.
- Grouped attention currently performs repeated frame-level `cat` and CPU-index
  construction. Chunk windows are contiguous and should use cached slices.
- Reconstructing epsilon with `new_tensor(...).item()` introduces a GPU-to-host
  synchronization per block and slightly changes epsilon in low precision. Use the
  Python float directly.

### Decision for our private port

Use Saganaki22 as an Apache-2.0 attributed behavioral reference, not as the codebase
to install or lightly fork. Our implementation keeps the earlier `ContextVar` layout
design and adds:

- strict adapter inventory with zero silently skipped tensors;
- compact Q/K/V offset patches instead of block-diagonal LoRA matrices;
- a Comfy-managed resident weight object for 96 GB, plus an explicit releasable
  pinned double-buffer streaming fallback;
- bounded per-model caches keyed by the full device and geometry;
- path containment and safetensors-only validation;
- early release/view use for Q/K/V and contiguous-slice grouped attention;
- full-cover, token-refiner, dual-GPU, 50-block, short-conv, gate, text-state, and
  official-module parity tests before accepting real-weight output.

## Private alpha implementation snapshot

An independent implementation now exists under
`code/ComfyUI-Kirei-VDN-H3/`. A private clone is installed under ComfyUI custom nodes
for integration testing; ComfyUI core remains unmodified.

Implemented:

- native reversible `ModelPatcher` integration and collision checks;
- strict `PackedLayout` conversion through a scoped `ContextVar`;
- grouped SDPA and FlexAttention window paths with full-cover gate correction;
- VDN/Sana rules, bidirectional scans, short convolution, bridge/gates/text state;
- strict safetensors/spec inventory and realpath containment;
- correct token-refiner mapping and factorized native LoRA patches with Q/K/V row offsets;
- runtime curve-AdaLN injection for pruned H3 without materializing full-width deltas;
- per-model resident/stream branch-weight ownership and explicit release node;
- bounded per-model caches keyed by full CUDA device; no global GPU tensor caches.

Verification performed on 2026-09-03:

- complete synthetic suite after post-push review: **55 passed**;
- Python compile/import: passed;
- real plugin entrypoint import against the installed ComfyUI tree: passed;
- nodes exposed: `KireiApplyVDNH3Alpha`, `KireiReleaseVDNH3Weights`;
- RTX PRO 6000 Flex versus grouped CUDA smoke test: BF16 max absolute error
  `0.00390625`, `allclose=True`;
- cache lifecycle probe: one entry before release, zero after release.
- released `stage-dmd-step-250` inventory and strict safetensors loading: passed;
- real H3 INT8 application: 520 low-rank terms over 208 weight targets, 51 curve
  targets, and 50 branch blocks;
- CUDA materialization probe of a quantized fused QKV with stacked offset LoRAs: passed;
- complete 4-step 608×352, 22-frame generation with video and audio decode: passed in
  37 seconds on the RTX PRO 6000 launcher path.

Remaining gates: direct numerical/visual parity against the official implementation,
longer-resolution stress tests, determinism, memory profiling, and output-quality
evaluation. The project remains an unsupported alpha.
