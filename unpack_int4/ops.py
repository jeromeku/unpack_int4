from importlib.metadata import version

import torch
from torch import Tensor

library_name = "unpack_int4"


def register_custom_op(name):
    def decorator(func):
        if version("torch") >= "2.4.0.dev":
            return torch.library.register_fake(f"{name}")(func)
        else:
            return torch.library.impl_abstract(f"{name}")(func)

    return decorator


def unpack_int4_packed(packed_w: Tensor, innerKTiles: int) -> Tensor:
    """
    Unpacks weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`.

    Assumes that the packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `innerKTiles = 2 | 4 | 8`"

    Args:
        packed_w: torch.tensor: 4D tensor with shape (N / 8) x (K / (innerKTiles * 16)) x 32 x innerKTiles, dtype is torch.int32
        innerKTiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.int32

    """
    return torch.ops.unpack_int4.unpack_int4_packed.default(
        packed_w=packed_w, innerKTiles=innerKTiles
    )


@register_custom_op(f"{library_name}::unpack_int4_packed")
def _(packed_w: Tensor, innerKTiles: int) -> Tensor:
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 42d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        innerKTiles == 2 or innerKTiles == 4 or innerKTiles == 8,
        lambda: "innerKTiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == innerKTiles / 2,
        lambda: "packed weight must have innerKTiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * innerKTiles * 16

    return torch.empty((N, K), dtype=torch.int32, device=packed_w.device)
