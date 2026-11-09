import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.cuda.set_device(rank)

        tensor = torch.ones(1, device=f"cuda:{rank}") * rank
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        expected = sum(range(world_size))
        result = tensor.item()
        status = "PASSED" if result == expected else "FAILED"
        print(
            f"[rank {rank}] all_reduce result={result} "
            f"expected={expected} -> {status}"
        )
        if result != expected:
            raise RuntimeError(f"Incorrect NCCL all_reduce result on rank {rank}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Need >=2 GPUs for a distributed test, found {world_size}"
    print(f"Running NCCL all_reduce smoke test across {world_size} GPUs")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
