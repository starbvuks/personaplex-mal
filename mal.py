"""Equation-based speech MAL adapter. See docs/experiments.md for adaptations.

Interface parameters are global learned parameters; FastState is episode state.
The analytic inner gradient remains differentiable for future outer-loop training.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'personaplex-mal.v1'

@dataclass
class FastState:
    w1: torch.Tensor
    w2: torch.Tensor
    u1: torch.Tensor
    u2: torch.Tensor
    writes: int = 0

    def payload(self):
        return {k: v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v
                for k,v in vars(self).items()}

    def detached(self):
        return FastState(*(v.detach() if isinstance(v,torch.Tensor) else v for v in vars(self).values()))


def read_memory(state, query):
    return F.linear(F.silu(F.linear(query,state.w1)), state.w2)


def update_memory(state, key, value, forget, momentum, learning_rate):
    """Single-token .5 * sum squared error; exact gradient through both MLP layers."""
    pre=F.linear(key,state.w1)
    sig=pre.sigmoid()
    activation=pre*sig
    error=F.linear(activation,state.w2)-value
    g2=torch.outer(error,activation)
    g1=torch.outer((state.w2.T@error)*(sig+pre*sig*(1-sig)),key)
    u1=momentum*state.u1-learning_rate*g1
    u2=momentum*state.u2-learning_rate*g2
    return FastState((1-forget)*state.w1+u1,(1-forget)*state.w2+u2,u1,u2,state.writes+1)


class MALBranch(nn.Module):
    def __init__(self, host_dim, memory_dim=128, persistent_tokens=8, policy=None):
        super().__init__()
        self.config={'host_dim':host_dim,'memory_dim':memory_dim,'persistent_tokens':persistent_tokens}
        from memory_policy import validate_policy
        self.policy=validate_policy({} if policy is None else policy)
        self.write_mask=None
        self.write_plan=[]
        self.write_observer=None
        self.down=nn.Linear(host_dim,memory_dim,bias=False)
        self.up=nn.Linear(memory_dim,host_dim,bias=False)
        self.query=nn.Linear(memory_dim,memory_dim,bias=False)
        self.key=nn.Linear(memory_dim,memory_dim,bias=False)
        self.value=nn.Linear(memory_dim,memory_dim,bias=False)
        self.rates=nn.Linear(memory_dim,3)
        # Conservative engineering initialization, not learned gating parameters.
        nn.init.zeros_(self.rates.weight)
        with torch.no_grad(): self.rates.bias.copy_(torch.tensor([-8.,2.,-5.]))
        self.persistent=nn.Parameter(torch.randn(persistent_tokens,memory_dim)*.02)
        self.initial_w1=nn.Parameter(torch.randn(memory_dim,memory_dim)*memory_dim**-.5)
        self.initial_w2=nn.Parameter(torch.randn(memory_dim,memory_dim)*memory_dim**-.5)
        self.alpha=nn.Parameter(torch.zeros(()))
        self.fast=None
        self.frozen=False
        self.prefix_pending=True
        self.observer=None

    def initial_state(self):
        return FastState(self.initial_w1,self.initial_w2,torch.zeros_like(self.initial_w1),torch.zeros_like(self.initial_w2))

    def begin_episode(self, reset=False):
        if self.frozen: raise RuntimeError('Frozen history cannot begin a write episode')
        if reset: self.fast=None
        self.prefix_pending=True

    def freeze_state(self):
        self.fast=(self.fast or self.initial_state()).detached()
        self.frozen=True
        self.prefix_pending=False
        self.requires_grad_(False)

    def _write(self,state,z):
        key=F.normalize(self.key(z),dim=-1,eps=1e-6)
        value=self.value(z)
        forget,momentum,rate=self.rates(z).sigmoid().unbind(-1)
        if self.policy.get('rates_mode')=='manual':
            forget=value.new_tensor(self.policy.get('forget',.0003))
            momentum=value.new_tensor(self.policy.get('momentum',.88))
            rate=value.new_tensor(self.policy.get('learning_rate',.0067))
        result=update_memory(state,key,value,forget,momentum,rate)
        proposed_change=(torch.stack([(result.w1-state.w1).norm(),(result.w2-state.w2).norm()]).norm()
                         if self.write_observer else None)
        cap=self.policy.get('update_max_norm',0)
        if cap:
            u1_scale=(cap/result.u1.square().sum().clamp_min(1e-24).sqrt()).clamp(max=1)
            u2_scale=(cap/result.u2.square().sum().clamp_min(1e-24).sqrt()).clamp(max=1)
            u1=result.u1*u1_scale
            u2=result.u2*u2_scale
            result=FastState((1-forget)*state.w1+u1,(1-forget)*state.w2+u2,u1,u2,result.writes)
        # Unlike the legacy momentum cap, this constrains the complete change,
        # including forgetting. Scale the stored momentum by the same factor so
        # a clipped impulse cannot accumulate invisibly for the next write.
        scale=value.new_ones(())
        cap=self.policy.get('state_update_max_norm',0)
        if cap:
            d1=result.w1-state.w1; d2=result.w2-state.w2
            change=d1.square().sum().add(d2.square().sum()).clamp_min(1e-24).sqrt()
            scale=(cap/change).clamp(max=1)
            result=FastState(state.w1+scale*d1,state.w2+scale*d2,
                             result.u1*scale,result.u2*scale,result.writes)
        if self.write_observer:
            self.write_observer(state,result,{'forget':forget,'momentum':momentum,'learning_rate':rate,
                'key_norm':key.norm(),'value_norm':value.norm(),'feature_norm':z.norm(),
                'proposed_state_change_norm':proposed_change,'state_update_scale':scale,
                'applied_state_change_norm':torch.stack([(result.w1-state.w1).norm(),(result.w2-state.w2).norm()]).norm()})
        return result

    def forward(self,x,write=False):
        if x.ndim!=3 or x.shape[0]!=1: raise ValueError('MAL experiment supports batch-one [1,time,dim]')
        if write and self.frozen: raise RuntimeError('Frozen history rejects writes')
        write=write and self.policy.get('write_enabled',True)
        state=self.fast if self.fast is not None else self.initial_state()
        features=x.to(self.down.weight.dtype)
        if self.policy.get('feature_rms_norm',False):
            # Clamp the mean square BEFORE sqrt, not the rms after it. The two are
            # value-identical in the forward pass, but sqrt()'s backward is
            # grad/(2*sqrt(x)); when a frame's mean square is exactly 0 the old
            # clamp-after-sqrt multiplied that inf by the clamp's zero gradient and
            # produced NaN in the lowest MAL branch (detect_anomaly: SqrtBackward0).
            features=features/features.square().mean(dim=-1,keepdim=True).clamp_min(1e-12).sqrt()
        z=self.down(features)[0]
        if write and self.prefix_pending:
            for index,p in enumerate(self.persistent if self.policy.get('write_prefix',True) else []):
                self.write_position=-(index+1)
                state=self._write(state,p)
            self.prefix_pending=False
        reads=[]; queries=[]
        for position,token in enumerate(z):
            q=F.normalize(self.query(token),dim=-1,eps=1e-6)
            if self.observer: queries.append(q)
            reads.append(read_memory(state,q))  # read BEFORE current-token write
            if write and (self.write_mask is None or self.write_mask[position]):
                self.write_position=position
                state=self._write(state,token)
        read=torch.stack(reads).unsqueeze(0)
        delta=self.up(read)*self.alpha.tanh()*self.policy.get('read_scale',1.)
        proposed_delta=delta
        cap=self.policy.get('residual_max_ratio',0)
        if cap:
            maximum=x.float().norm(dim=-1,keepdim=True)*cap
            denom=delta.square().sum(dim=-1,keepdim=True).clamp_min(1e-24).sqrt()
            delta=delta*(maximum/denom).clamp(max=1)
        if write: self.fast=state
        if self.observer:
            self.observer('input',x)
            self.observer('query',torch.stack(queries))
            self.observer('read',read)
            self.observer('residual_proposed',proposed_delta)
            self.observer('residual',delta)
            self.observer('status',{'writes':state.writes,'frozen':self.frozen,'write_enabled':write,
                                    'gate':float(self.alpha.detach().tanh().cpu())})
        return x+delta.to(x.dtype)


def tensor_hash(mapping):
    digest=hashlib.sha256()
    for key,value in sorted(mapping.items()):
        digest.update(key.encode())
        if isinstance(value,torch.Tensor):
            t=value.detach().cpu().contiguous()
            digest.update(str((tuple(t.shape),str(t.dtype))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        else: digest.update(str(value).encode())
    return digest.hexdigest()


def save_checkpoint(path,branches,metadata):
    if not all(b.frozen for b in branches): raise ValueError('Freeze all branches before persistence')
    interface={k:v.detach().cpu() for k,v in branches.state_dict().items()}
    states=[b.fast.payload() for b in branches]
    kv_states=[]
    for b in branches:
        km=getattr(b,'kv_memory',None)
        kv_states.append(None if km is None else {'config':km.config_dict(), **km.bank_dict()})
    payload={'format':FORMAT,'configs':[{**b.config,'policy':b.policy} for b in branches], 'interface':interface,
             'layer_indices':[getattr(b,'layer_index',i) for i,b in enumerate(branches)],
             'states':states,'kv_states':kv_states,'interface_sha256':tensor_hash(interface),
             'state_sha256':[tensor_hash(s) for s in states],
             'metadata':{**metadata,'training_status':metadata.get('training_status','untrained')}}
    path=Path(path); temp=path.with_suffix('.tmp')
    torch.save(payload,temp); temp.replace(path)
    return payload


def load_checkpoint(path,expected_revision=None,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!=FORMAT: raise ValueError('Unsupported MAL checkpoint format; static adapters are not MAL')
    if tensor_hash(payload['interface'])!=payload['interface_sha256']: raise ValueError('Interface hash mismatch')
    if len(payload['configs'])!=len(payload['states']): raise ValueError('State count mismatch')
    if [tensor_hash(s) for s in payload['states']]!=payload['state_sha256']: raise ValueError('State hash mismatch')
    meta=payload['metadata']
    if expected_revision and meta.get('model_revision')!=expected_revision: raise ValueError('Model revision mismatch')
    kv_states=payload.get('kv_states',[None]*len(payload['configs']))
    branches=nn.ModuleList([MALBranch(**c) for c in payload['configs']])
    for branch,kv in zip(branches,kv_states):
        if kv is not None:
            from kv_read import KVMemory
            branch.add_module('kv_memory',KVMemory(**kv['config']))
    branches.load_state_dict(payload['interface'],strict=True)
    branches.to(device)
    for index,(branch,state,kv) in enumerate(zip(branches,payload['states'],kv_states)):
        branch.layer_index=payload.get('layer_indices',list(range(len(branches))))[index]
        branch.fast=FastState(**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in state.items()})
        branch.freeze_state()
        if kv is not None:
            branch.kv_memory.load_bank(kv)
    return branches,{**meta,'interface_sha256':payload['interface_sha256'],'state_sha256':payload['state_sha256']}


def install_branches(lm,branches,write=False):
    layers=lm.transformer.layers
    indices=[getattr(b,'layer_index',i) for i,b in enumerate(branches)]
    if len(set(indices))!=len(indices) or any(type(i) is not int or i<0 or i>=len(layers) for i in indices):
        raise ValueError('Invalid or duplicate temporal layer index')
    if len(branches)!=len(layers) and any(not hasattr(b,'layer_index') for b in branches):
        raise ValueError('Sparse branches require explicit layer indices')
    handles=[]
    for index,branch in zip(indices,branches):
        layer=layers[index]
        def hook(module,args,b=branch): return (b(args[0],write=write),*args[1:])
        handles.append(layer.register_forward_pre_hook(hook))
    return handles
