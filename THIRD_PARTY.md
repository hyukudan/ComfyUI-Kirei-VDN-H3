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

This project is an independent ComfyUI implementation. Adapted source and algorithms
retain the applicable copyright and license notices.
