###Q3: allreduce###
###implement ring all-reduce with isend/irecv; built-in collectives are not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist
import torch.nn.functional as F

def reduce_scatter(chunks, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    send_idx = rank
    for _ in range(world - 1):
        recv_idx = (send_idx - 1) % world
        recv_buf = torch.empty_like(chunks[recv_idx])

        s = dist.isend(chunks[send_idx], dst=right) # send to right
        r = dist.irecv(recv_buf, src=left)          # recv from left
        s.wait()
        r.wait()

        chunks[recv_idx] += recv_buf
        send_idx = recv_idx  # counter-clockwise

    # rank r owns the fully reduced chunk (r + 1) % world
    return

def all_gather(chunks, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    send_idx = (rank + 1) % world
    for _ in range(world - 1):
        recv_idx = (send_idx - 1) % world

        s = dist.isend(chunks[send_idx], dst=right)
        r = dist.irecv(chunks[recv_idx], src=left)
        s.wait()
        r.wait()

        send_idx = recv_idx

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
    pad_size = chunk * world - n
    padded_flat = F.pad(flat, (0, pad_size), mode='constant', value=0)
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    reduce_scatter(chunks, world, rank, left, right)
    all_gather(chunks, world, rank, left, right)

    # stitch & unpad: the reduced data lives in padded_flat, not flat
    padded_flat /= world
    tensor.view(-1).copy_(padded_flat[:n])
    return