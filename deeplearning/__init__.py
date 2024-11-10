def check_torch_installation():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is not installed. Please install it from https://pytorch.org/get-started/locally/ "
            "based on your system configuration."
        ) from e
