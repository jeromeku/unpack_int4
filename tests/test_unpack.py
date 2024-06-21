import itertools

import pytest
import torch
from torch.testing._internal.optests import opcheck

from unpack_int4.ops import unpack_int4_packed

kTileSizeN = 8
kTileSizeK = 16

SHAPES = [
    (4096, 4096),
    # Llama 2 GEMM shapes
    (4096, 11008),
    (11008, 4096),
    # Llama 3 GEMM shapes
    (4096, 14336),
    (14336, 4096),
]
INNERKTILES = [2, 4, 8]

TEST_CONFIGS = list(itertools.product(SHAPES, INNERKTILES))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS, ids=str)
def test_int4_unpack_correctness(shape, innerKTiles):
    N, K = shape
    assert K % (innerKTiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, innerKTiles)
    unpacked = unpack_int4_packed(packed_w, innerKTiles)
    assert torch.allclose(t, unpacked)


@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS, ids=str)
def test_unpack_int4_op(shape, innerKTiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        # "test_aot_dispatch_dynamic",
    ]
    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, innerKTiles)

    opcheck(
        torch.ops.unpack_int4.unpack_int4_packed,
        (packed_w, innerKTiles),
        test_utils=test_utils,
    )
