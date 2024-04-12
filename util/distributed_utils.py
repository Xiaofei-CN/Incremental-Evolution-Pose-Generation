import os

import torch
import torch.distributed as dist
import numpy as np
import random
import torch.backends.cudnn as cudnn
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, string_classes
else:
    import collections.abc as container_abcs
    string_classes = str
def init_distributed_mode(args):
    seed = 1
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.lr *= args.world_size
    args.distributed = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch
def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def init_dist(local_rank, backend='nccl', **kwargs):
    r"""Initialize distributed training"""
    if dist.is_available():
        if dist.is_initialized():
            return torch.cuda.current_device()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://', **kwargs)

def init_cudnn(deterministic, benchmark):
    r"""Initialize the cudnn module. The two things to consider is whether to
    use cudnn benchmark and whether to use cudnn deterministic. If cudnn
    benchmark is set, then the cudnn deterministic is automatically false.

    Args:
        deterministic (bool): Whether to use cudnn deterministic.
        benchmark (bool): Whether to use cudnn benchmark.
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    print('cudnn benchmark: {}'.format(benchmark))
    print('cudnn deterministic: {}'.format(deterministic))

def to_device(data, device):
    r"""Move all tensors inside data to device.

    Args:
        data (dict, list, or tensor): Input data.
        device (str): 'cpu' or 'cuda'.
    """
    assert device in ['cpu', 'cuda']
    if isinstance(data, torch.Tensor):
        data = data.to(torch.device(device))
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_device(d, device) for d in data]
    else:
        return data


def to_cuda(data):
    r"""Move all tensors inside data to gpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cuda')
def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value
