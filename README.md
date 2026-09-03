# ComfyUI Kirei VDN-H3 — Private Alpha

> **ALPHA SOFTWARE — ACTIVE WORK IN PROGRESS**
>
> `ComfyUI-Kirei-VDN-H3` is an experimental, private, correctness-first port of
> VideoDeltaNet for ComfyUI's native MiniMax H3 implementation. It is incomplete,
> its API and checkpoint format may change without notice, and it has not yet been
> validated end-to-end with the released VDN weights. Do not use it for production
> work or install it into an important ComfyUI environment.

## Current status

- Native ComfyUI model-patch architecture: implemented for alpha testing.
- Windowed softmax and VDN linear branch: implemented.
- Strict checkpoint and adapter mapping: implemented.
- Synthetic CPU tests: 48 passing.
- RTX PRO 6000 CUDA validation: Flex/grouped smoke parity passed.
- Real VDN checkpoint validation: blocked pending model-license authorization.
- Output-quality and performance claims: none yet.

Development happens in this isolated directory. The live ComfyUI checkout at
`D:\comfyui\ComfyUI` must not be modified until the synthetic and CUDA test gates pass.

## Goals

- Faithful VDN-H3 math on ComfyUI's native MiniMax H3 model.
- Reversible `ModelPatcher` integration without ComfyUI core edits.
- Strict, explicit adapter mapping with no silently ignored tensors.
- Compact fused-QKV LoRA handling without block-diagonal zero matrices.
- Memory ownership and cache eviction that cooperate with ComfyUI.
- A fast RTX PRO 6000 path with a portable correctness fallback.

## Non-goals for this alpha

- Bundling or automatically downloading model weights.
- Claiming parity before comparison with the official implementation.
- Reproducing the official 8×B200 headline benchmark on one workstation GPU.
- Maintaining backward compatibility while the internal design is still changing.

## Safety and licensing

The code references Apache-2.0 implementations; preserve the attribution recorded in
`THIRD_PARTY.md` and any applicable source notices. Model weights are not bundled and
remain governed by their own licenses. The published MiniMax H3/VDN model materials
state territorial restrictions that include the European Union; obtain appropriate
authorization before downloading or using those weights in Spain. This note is an
engineering precaution, not legal advice.

## Installation

There is intentionally no supported installation procedure yet. When the alpha passes
its correctness and lifecycle tests, it will first be tested as a copied custom node in
an isolated ComfyUI environment—not in the active installation.

## Alpha verification snapshot

Validated on 2026-09-03 with Python 3.12.9, PyTorch 2.12.0+cu130 and an NVIDIA RTX
PRO 6000 Blackwell Workstation Edition:

- `48 passed` in the complete synthetic pytest suite.
- Plugin entrypoint imports against the installed ComfyUI source.
- Registered nodes: `KireiApplyVDNH3Alpha` and `KireiReleaseVDNH3Weights`.
- CUDA FlexAttention agrees with grouped SDPA within BF16 tolerance
  (`max_abs_error=0.00390625`).
- Per-model Flex cache contains one entry during the probe and zero after `release()`.

These checks do **not** establish real-checkpoint visual parity, generation quality,
or production stability.
