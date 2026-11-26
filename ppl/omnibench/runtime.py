"""Small CUDA/Ascend/CPU adapter; training math is shared across devices."""
from contextlib import nullcontext
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from .settings import CONFIG


def device_type():
    requested=os.environ.get('OMNIBENCH_DEVICE',CONFIG.get('runtime',{}).get('device','auto'))
    if requested=='cpu':return 'cpu'
    if requested in ['auto','cuda'] and torch.cuda.is_available():return 'cuda'
    if requested in ['auto','npu']:
        try:
            import torch_npu
            if torch.npu.is_available():return 'npu'
        except ImportError:
            pass
    if requested=='auto':return 'cpu'
    raise RuntimeError(f'Requested backend {requested} is unavailable; run the doctor command')


def get_device(index=None):
    kind=device_type();index=int(os.environ.get('LOCAL_RANK',0)) if index is None else index
    device=torch.device('cpu' if kind=='cpu' else f'{kind}:{index}')
    if kind!='cpu':getattr(torch,kind).set_device(device)
    return device


def model_dtype():
    value=CONFIG.get('runtime',{}).get('dtype','bfloat16')
    if device_type()=='cpu':return torch.float32
    if value=='float32':return torch.float32
    if value=='bfloat16':
        if device_type()=='cuda' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This GPU lacks BF16 support; set runtime.dtype=float32 and lower micro batches')
        return torch.bfloat16
    raise ValueError('Supported model dtypes: bfloat16, float32')


def autocast(device):
    kind=torch.device(device).type
    return torch.autocast(kind,dtype=model_dtype()) if kind!='cpu' and model_dtype()!=torch.float32 else nullcontext()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    kind=device_type()
    if kind!='cpu':getattr(torch,kind).manual_seed_all(seed)


def setup_distributed(device):
    if int(os.environ.get('WORLD_SIZE',1))>1:
        dist.init_process_group({'cuda':'nccl','npu':'hccl','cpu':'gloo'}[device.type])


def barrier():
    if dist.is_initialized():dist.barrier()


def reduce_sum(tensor):
    if dist.is_initialized():dist.all_reduce(tensor)


def broadcast(tensor):
    if dist.is_initialized():dist.broadcast(tensor,src=0)


def close_distributed():
    if dist.is_initialized():dist.destroy_process_group()


def peak_memory():
    kind=device_type()
    return int(getattr(torch,kind).max_memory_allocated()) if kind!='cpu' else 0


def capture_rng(device):
    state={'cpu':torch.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state()}
    if device.type!='cpu':state['accelerator']=getattr(torch,device.type).get_rng_state(device)
    return state


def restore_rng(state,device):
    torch.set_rng_state(state['cpu'])
    if 'python' in state:random.setstate(state['python'])
    if 'numpy' in state:np.random.set_state(state['numpy'])
    if device.type!='cpu':getattr(torch,device.type).set_rng_state(state['accelerator'],device)


def iterator(loader,sampler,start_step=0):
    epoch,offset=divmod(start_step,len(loader))
    while True:
        if sampler is not None:sampler.set_epoch(epoch)
        for index,indices in enumerate(loader.batch_sampler):
            if index>=offset:yield loader.collate_fn([loader.dataset[i] for i in indices])
        epoch+=1;offset=0
