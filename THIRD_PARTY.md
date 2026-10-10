# Third-party references

- OpenVDN/vdn-minimax-h3 — Apache-2.0 reference implementation.
  - The bidirectional inference scan/preallocation strategy, five-tap Triton temporal
    convolution, decomposed varlen attention plan and opt-in FP8 projection approach
    in this project are adapted from or informed by the OpenVDN implementation.
  - The FP8 path retains OpenVDN's architecture-dependent scaling policy: per-tensor
    activation/weight scales on SM100 and rowwise activation plus per-output-channel
    weight scales on earlier supported NVIDIA architectures.
- Saganaki22/ComfyUI-VDN-H3 — Apache-2.0 ComfyUI integration reference.
  - Used for compatibility comparison around native MiniMax-H3 target mapping,
    grouped/Flex attention, adapter application and INT8/ConvRot integration.
- xmarre/ComfyUI-VDN-H3-Plus — Apache-2.0 compatibility reference for the published
  AdaLN affine companion format, curve-table identity checks and attention buffer
  lifetimes. The local fallback reconstructs the embedding grid and retains Kirei's
  existing adapter execution; it does not copy Plus's affine projection runtime.
  The reduced pinned OpenVDN source-oracle tests are adapted from Plus and retain
  their Apache-2.0 attribution in `tests/test_official_oracle.py`.
- Speach1sdef178/ComfyUI-VELA-H3 — Apache-2.0 research reference for grouped-window
  batching and selective exact/approximate attention. No VELA approximation policy,
  environment overrides or retained research runtime is included in Kirei.

This project is an independent ComfyUI implementation. Adapted source and algorithms
retain the applicable copyright and license notices.
