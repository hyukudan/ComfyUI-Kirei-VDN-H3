"""Keep current-ComfyUI integration tests usable on CPU-only runners."""

import torch

if not torch.cuda.is_available():
    try:
        from comfy.cli_args import args
    except ImportError:
        pass
    else:
        args.cpu = True
