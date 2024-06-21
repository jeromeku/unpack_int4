## Description

CUDA custom op to unpack weights that have been packed with `torch.ops.aten._convert_weight_to_int4pack` for use with `torch.ops.aten._weight_int4pack_mm`.  

Currently there is only a packing function that permutes and prepacks the weights in tensor-core format.  However, there is no equivalent unpacking function that reorders the weights back to the original logical layout.

## Motivation
Unpacking the weights is needed when the weights have been packed for inference.  For the workload transitions from memory-bound to compute-bound (i.e., context length growth during decoding), users might wish to switch to a different kernel implementation that is more performant in this regime.  

In order to do this, the weights need to be unpacked from the packed format.  Alternative would be to store 2 copies of the weights -- one packed, one in logical format -- but this is clearly not ideal given memory constraints.

## Features
Custom CUDA implementation along with glue functions to register as a custom op that can be called with `torch.compile`.

## Tests
See `tests/test_unpack.py` for both correctness as well as for correct custom op registration.  Note that currently the custom op tests pass except for `test_aot_dispatch_dynamic`.

## Benchmarks
See `benchmarks/unpack_bench.py` for benchmark comparing the cuda extension vs. compiled cuda vs. a reference (compiled) torch-native implementation.

To run against a battery of shapes:
```
python benchmarks/unpack_bench.py
```

Run against a single shape:
```
python benchmarks/unpack_bench.py --shape 8192 8192
```
