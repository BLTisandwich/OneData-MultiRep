"""Equivalent PyTorch DeltaNet chunk algebra without CUDA fused kernels.

Adapted from Transformers 5.3 Qwen3.5 torch_chunk_gated_delta_rule (Apache-2.0).
The unit lower triangular inverse is a finite nilpotent geometric series,
evaluated by repeated squaring instead of 63 sequential in-place row updates.
No model weights, decay equations or attention semantics are changed.
"""
import torch
import torch.nn.functional as F

def chunk_delta(query,key,value,g,beta,chunk_size=32,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False):
    # Explicit .float() does not prevent an enclosing autocast context from
    # recasting matrix products. DeltaNet's triangular inverse/state algebra
    # needs FP32 even when projections and the transformer use bfloat16.
    with torch.autocast(device_type=query.device.type,enabled=False):
        return _chunk_delta_fp32(query,key,value,g,beta,chunk_size,initial_state,output_final_state,use_qk_l2norm_in_kernel)

def _chunk_delta_fp32(query,key,value,g,beta,chunk_size=32,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False):
    dtype=query.dtype
    query,key=query.float(),key.float()
    if use_qk_l2norm_in_kernel:
        query=query*torch.rsqrt((query*query).sum(-1,keepdim=True)+1e-6)
        key=key*torch.rsqrt((key*key).sum(-1,keepdim=True)+1e-6)
    query,key,value,beta,g=[x.transpose(1,2).contiguous().float() for x in (query,key,value,beta,g)]
    b,h,length,kdim=key.shape;vdim=value.shape[-1]
    pad=(chunk_size-length%chunk_size)%chunk_size
    query,key,value=[F.pad(x,(0,0,0,pad)) for x in (query,key,value)]
    beta,g=[F.pad(x,(0,pad)) for x in (beta,g)]
    query=query/(kdim**.5)
    vb=value*beta.unsqueeze(-1);kb=key*beta.unsqueeze(-1)
    query,key,value,kb,vb=[x.reshape(b,h,-1,chunk_size,x.shape[-1]) for x in (query,key,value,kb,vb)]
    g=g.reshape(b,h,-1,chunk_size).cumsum(-1)
    decay=(g.unsqueeze(-1)-g.unsqueeze(-2)).tril().exp().tril()
    power=(-(kb@key.transpose(-1,-2))*decay).tril(diagonal=-1)
    eye=torch.eye(chunk_size,device=query.device,dtype=torch.float32)
    inverse=eye+power
    exponent=2
    while exponent<chunk_size:
        power=power@power
        inverse=inverse+inverse@power
        exponent*=2
    value=inverse@vb
    kc=inverse@(kb*g.exp().unsqueeze(-1))
    state=torch.zeros(b,h,kdim,vdim,device=value.device,dtype=value.dtype) if initial_state is None else initial_state.to(value)
    outputs=[]
    for i in range((length+pad)//chunk_size):
        q,k,v=query[:,:,i],key[:,:,i],value[:,:,i]
        attn=(q@k.transpose(-1,-2)*decay[:,:,i]).tril()
        vn=v-kc[:,:,i]@state
        outputs.append((q*g[:,:,i,:,None].exp())@state+attn@vn)
        state=state*g[:,:,i,-1,None,None].exp()+(k*(g[:,:,i,-1,None]-g[:,:,i]).exp()[...,None]).transpose(-1,-2)@vn
    output=torch.stack(outputs,dim=2).reshape(b,h,-1,vdim)[:,:,:length].transpose(1,2).contiguous().to(dtype)
    return output,state if output_final_state else None

def install(model):
    count=0
    for module in model.modules():
        if type(module).__name__=='Qwen3_5GatedDeltaNet':
            module.chunk_gated_delta_rule=chunk_delta;count+=1
    return count

def stable_attention(model):
    """Keep attention/state algebra and its projections FP32 on both backbones.

    Frozen FFNs/embeddings remain bfloat16. Outputs return to the incoming dtype.
    This rule applies equally to Llama attention and Qwen hybrid attention.
    """
    def cast_output(value,dtype):
        if torch.is_tensor(value) and value.is_floating_point():return value.to(dtype)
        if isinstance(value,tuple):return tuple(cast_output(v,dtype) for v in value)
        return value
    count=0
    for module in model.modules():
        if type(module).__name__ not in ['Qwen3_5GatedDeltaNet','Qwen3_5Attention','LlamaAttention','Qwen2Attention']:continue
        module.float()
        original=module.forward
        def make_forward(original):
            def forward(hidden_states,*args,**kwargs):
                dtype=hidden_states.dtype
                with torch.autocast(device_type=hidden_states.device.type,enabled=False):
                    result=original(hidden_states.float(),*args,**kwargs)
                return cast_output(result,dtype)
            return forward
        module.forward=make_forward(original);count+=1
    return count
