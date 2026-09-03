# ComfyUI Kirei VDN-H3 — Private Alpha

> **ALPHA SOFTWARE — ACTIVE WORK IN PROGRESS**
>
> `ComfyUI-Kirei-VDN-H3` is an experimental, private, correctness-first port of
> VideoDeltaNet for ComfyUI's native MiniMax H3 implementation. It is incomplete,
> its API and checkpoint format may change without notice. One short real-weight
> smoke video has passed, but parity and quality are not established. Do not use it for production
> work or install it into an important ComfyUI environment.

## Current status

- Native ComfyUI model-patch architecture: implemented for alpha testing.
- Windowed softmax and VDN linear branch: implemented.
- Strict checkpoint and adapter mapping: implemented.
- Synthetic CPU tests: 55 passing.
- RTX PRO 6000 CUDA validation: Flex/grouped smoke parity passed.
- Real VDN checkpoint: complete inventory and strict structural loading passed.
- Factorized native ComfyUI LoRA path: real checkpoint application passed.
- Real 608×352, 22-frame video+audio smoke generation: passed on RTX PRO 6000.
- Output-quality and performance claims: none yet.

Development happens in this isolated directory. A private clone is installed as a
custom node for integration testing without modifying ComfyUI core.

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

- `55 passed` in the complete synthetic pytest suite.
- Plugin entrypoint imports against the installed ComfyUI source.
- Registered nodes: `KireiApplyVDNH3Alpha` and `KireiReleaseVDNH3Weights`.
- Released stage checkpoint inventory loads successfully from safetensors.
- Adapter A/B factors remain low-rank and use native ComfyUI offset patches; AdaLN
  curve adapters are injected at runtime for pruned H3 checkpoints.
- Real application registered 520 compact LoRA terms over 208 native weights, 51
  runtime curve-AdaLN targets, and 50 VDN branches.
- A 4-step, 608×352, 22-frame H3 INT8 smoke generated an H.264/AAC MP4 in 37 seconds.
- CUDA FlexAttention agrees with grouped SDPA within BF16 tolerance
  (`max_abs_error=0.00390625`).
- Per-model Flex cache contains one entry during the probe and zero after `release()`.

These checks do **not** establish parity with the official implementation, quality at
production durations/resolutions, numerical determinism, or production stability.

## Engineering report

The private feasibility study, upstream audit, hardware findings and phased validation
plan are maintained in [`docs/PORT_REPORT.md`](docs/PORT_REPORT.md).
