import torch


# All the code below is copied from @KeremTurgutlu
def reshape_packed(packed_tensor):
    inner_k_tiles = packed_tensor.size(-1) * 2
    return (
        packed_tensor.permute(0, 1, 3, 2)
        .reshape(
            packed_tensor.size(0),
            packed_tensor.size(1) * (inner_k_tiles // 2),
            packed_tensor.size(2),
            1,
        )
        .contiguous()
    )


def _unpack_shifting(packed_tensor):
    return [(packed_tensor >> (i * 4)) & 15 for i in range(8)]


@torch.compile(fullgraph=True)
def unpack_int4_32_pack_fast(packed_tensor, shape):
    reshaped_tensor = reshape_packed(packed_tensor)
    unpacked_tensors = _unpack_shifting(reshaped_tensor)

    # use torch.cat
    cat_tensors = [
        torch.cat(unpacked_tensors[i::4], dim=-1).view(-1, 8) for i in range(4)
    ]
    concatenated = torch.cat(cat_tensors, dim=-1)

    # # pre-allocate
    # concatenated = torch.empty(shape[0]*shape[1]//32, 32, device=reshaped_tensor.device, dtype=reshaped_tensor.dtype)
    # for i in range(4):
    #     concatenated[:,i*8:(i+1)*8] = torch.cat(unpacked_tensors[i::4], dim=-1).view(-1, 8)

    group_size = shape[1] // 32
    chunked_o = (
        concatenated.view(-1, 8)
        .unsqueeze(0)
        .view(concatenated.size(0) // 8, 8, -1)
        .unsqueeze(0)
    )
    res = (
        chunked_o.view(-1, group_size, chunked_o.size(2), chunked_o.size(3))
        .permute(0, 2, 1, 3)
        .reshape(shape)
    )

    return res


def unpack_int32(x):
    unpacked = torch.tensor([(x >> (i * 4) & 0xF) for i in range(8)])
    evens = unpacked[:4]
    odds = unpacked[4:]
    unpacked = torch.stack([evens, odds]).permute((1, 0)).flatten()
    return torch.tensor(unpacked)
