# Third-party references

- OpenVDN/vdn-minimax-h3 — Apache-2.0 reference implementation.
  - The inference scan/preallocation strategy, five-tap Triton temporal-convolution
    algorithm, and Blackwell decomposed-attention design in this project are adapted
    from the OpenVDN implementation.
  - Optimization audit/reference revision used during this pass:
    `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`.
- Saganaki22/ComfyUI-VDN-H3 — Apache-2.0 ComfyUI integration reference.
  - Used to compare native MiniMax-H3 target mapping, grouped/Flex attention behavior
    and INT8/ConvRot integration constraints.

This project is an independent private implementation. Any adapted source must retain
the applicable copyright and license notices.
