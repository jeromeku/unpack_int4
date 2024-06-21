import argparse

import torch
from tabulate import tabulate
from triton.testing import do_bench

import benchmarks.utils as utils
from unpack_int4.ops import unpack_int4_packed

kTileSizeN = 8
kTileSizeK = 16
shape = (8192, 8192)
# N, K = shape

SHAPES = [
    (8192, 8192),
    # Llama 2 GEMM shapes
    (4096, 11008),
    (11008, 4096),
    # Llama 3 GEMM shapes
    (4096, 14336),
    (14336, 4096),
]

unpack_cuda_compiled = torch.compile(unpack_int4_packed, mode="default", fullgraph=True)


def run_bench(shapes, kInnerTilesK, check=False):
    data = []
    for shape in shapes:
        t = torch.randint(0, 16, size=shape, dtype=torch.int, device="cuda")

        packed_ref = torch.ops.aten._convert_weight_to_int4pack(t, kInnerTilesK)

        if check:
            unpacked_cuda = unpack_int4_packed(packed_ref, kInnerTilesK)
            unpacked_ref = utils.unpack_int4_32_pack_fast(packed_ref, shape)

            if not torch.allclose(t, unpacked_cuda):
                print(f"Shape {shape}: unpack_int4_32_pack_fast check failed")

            if not torch.allclose(t, unpacked_ref):
                print(
                    f"Shape {shape}: unpack_cuda check failed",
                )

        t_ref = do_bench(lambda: utils.unpack_int4_32_pack_fast(packed_ref, shape))
        t_cuda = do_bench(lambda: unpack_int4_packed(packed_ref, kInnerTilesK))
        t_cuda_compiled = do_bench(
            lambda: unpack_cuda_compiled(packed_ref, kInnerTilesK)
        )

        data.append([shape, kInnerTilesK, t_ref, t_cuda, t_cuda_compiled])

    headers = [
        "Shape",
        "InnerKTiles",
        "unpack_int4_32_pack_fast",
        "custom cuda",
        "compiled cuda",
    ]
    table = tabulate(data, headers=headers, floatfmt=".4f")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(None, None),
        help="Shape to benchmark, e.g. 8192 8192",
    )
    parser.add_argument("--k_tiles", type=int, default=8, help="InnerKTiles: 2, 4, 8")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Additional sanity check that unpacking results in original tensor",
    )
    args = parser.parse_args()

    assert len(args.shape) == 2
    bench_all = args.shape == (None, None)
    if bench_all:
        run_bench(SHAPES, kInnerTilesK=args.k_tiles, check=args.check)
    else:
        run_bench([args.shape], kInnerTilesK=args.k_tiles, check=args.check)
