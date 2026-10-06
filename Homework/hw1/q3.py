###Q3: allreduce###
###implement ring all-reduce with isend/irecv; built-in collectives are not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist
import torch.nn.functional as F

def reduce_scatter(chunks, chunk_size, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    transfer_idx = rank  # start with transferring chunks[rank]
    for _ in range(world):
        recv_buf = torch.empty(chunk_size)
        recv_chunk_idx = (transfer_idx - 1) % world
        r = dist.irecv(recv_buf, src=left)
        r.wait()

        chunks[recv_chunk_idx] += recv_buf

        r = dist.isend(chunks[transfer_idx], dst=right)
        r.wait()    # wait for completion

        transfer_idx = (transfer_idx + 1) % world

    return

def all_gather(chunks, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    recv_idx = rank
    for _ in range(world):
        r = dist.irecv(chunks[recv_idx], src=left)
        r.wait()    # receive

        send_chunk_idx = (recv_idx + 1) % world
        r = dist.isend(chunks[send_chunk_idx], dst=right)
        r.wait()

        recv_idx = (recv_idx - 1) % world

    return

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce average using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    # TODO: likely need to call _unflatten_dense_tensors to unpack input

    pad_size = chunk * 3 - n
    padded_flat = F.pad(flat, (0, pad_size), mode='constant', value=0)  # modify this line and fill correct value into padded_flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    # TODO: call reduce_scatter
    reduce_scatter(chunks, chunk, world, rank, left, right)

    # TODO: call all_gather
    all_gather(chunks, world, rank, left, right)

    # stitch & unpad
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return