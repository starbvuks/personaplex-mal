"""Attention-coupled KV read path for MAL. Opt-in; the additive read stays default.

The write is NON-DENSE (hard sparse top-k slot write — an outer-product value
accumulation into at most k of S slots per token), unlike mal.update_memory's dense
attention-coupled gradient step. The read projects the written bank into the frozen
backbone's self-attention K/V through a trained, zero-initialized gate, by wrapping
`StreamingMultiheadAttention._complete_kv` from the outside. The vendored transformer
file is never edited and the 7B backbone stays frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F


class KVMemory(nn.Module):
    """Sparse-slot memory bank + attention K/V read with a trained gate.

    Per layer: a bank of S slots in memory_dim, written by a hard sparse top-k
    (non-dense) rule, read out into that layer's self-attention head space
    (num_heads x head_dim) through trained projections, gated by a zero-initialized
    scalar gate. Zero projected keys/values do not make the enabled attention
    wrapper neutral: the added position can still change attention normalization.
    """

    def __init__(self, memory_dim=128, num_slots=16, topk=4, num_heads=32, head_dim=128, lr=0.05):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.topk = min(topk, num_slots)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lr = lr
        self.write_key = nn.Linear(memory_dim, memory_dim, bias=False)
        self.write_value = nn.Linear(memory_dim, memory_dim, bias=False)
        self.slot_key = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.02)
        self.read_k = nn.Linear(memory_dim, num_heads * head_dim, bias=False)
        self.read_v = nn.Linear(memory_dim, num_heads * head_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        # Episode state: the written bank. Non-persistent; reset per episode.
        self.register_buffer('bank', torch.zeros(num_slots, memory_dim), persistent=False)
        self.enabled = False

    def reset(self):
        self.bank.zero_()
        self.last_write_idx = None

    def write(self, z, mask=None, addr=None):
        """Sparse slot write (non-dense). z: [T, memory_dim].

        - `mask=None`: legacy — every token writes to the top-k slots matching its own key.
        - `mask` given, `addr=None`: content-selective — only masked tokens are written,
          to the top-k slots of their mean content key.
        - `mask` given, `addr` given: content-addressed by token identity — the masked
          tokens' mean value is written to the single slot `addr % num_slots`, so
          token identities may collide in the same slot. Never a dense
          attention-coupled update.
        """
        if mask is not None:
            sel = z[mask]
            if sel.numel() == 0:
                return self.bank
        else:
            sel = z
        keys = F.normalize(self.write_key(sel), dim=-1)        # [n, m]
        values = self.write_value(sel)                          # [n, m]
        if addr is not None:
            # Content-addressed by token identity: the entity token id mod the number
            # of slots. collisions are possible; this is a fixed address, not a learned one.
            slot = int(addr) % self.num_slots
            self.last_write_idx = torch.tensor([slot], dtype=torch.long, device=z.device)
            update = torch.zeros_like(self.bank)
            update[slot] = values.mean(0)
        elif mask is None:
            sim = keys @ self.slot_key.T                        # [n, S]
            idx = sim.topk(self.topk, dim=-1).indices           # [n, k]
            self.last_write_idx = idx.detach()
            update = torch.zeros_like(self.bank)
            update = update.index_add(0, idx.reshape(-1), values.repeat_interleave(self.topk, dim=0))
        else:
            key_mean = keys.mean(0)                             # [m] content key
            sim = key_mean @ self.slot_key.T                    # [S]
            idx = sim.topk(self.topk, dim=-1).indices           # [k]
            self.last_write_idx = idx.detach()
            val = values.mean(0)                                # [m] content value
            update = torch.zeros_like(self.bank)
            update = update.index_add(0, idx, val.unsqueeze(0).repeat(self.topk, 1))
        self.bank = self.bank.detach() + self.lr * update
        return self.bank

    def bank_norm(self):
        return float(self.bank.norm())

    def slot_occupancy(self):
        """Number of slots with non-negligible content (a proxy for write sparsity)."""
        return int((self.bank.norm(dim=1) > 1e-6).sum().item())

    def summary(self):
        return self.bank.mean(0)                               # [m] slot mean

    def read_kv(self):
        """Project the written bank into attention K and V, scaled by the trained gate."""
        s = self.summary()
        K = self.read_k(s).view(self.num_heads, self.head_dim)
        V = self.read_v(s).view(self.num_heads, self.head_dim)
        g = torch.tanh(self.gate)                              # zero-init => no effect
        return K * g, V * g

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self):
        return self.parameter_count()

    def config_dict(self):
        """Constructor kwargs, for checkpoint round-trip."""
        return {'memory_dim': self.memory_dim, 'num_slots': self.num_slots, 'topk': self.topk,
                'num_heads': self.num_heads, 'head_dim': self.head_dim, 'lr': self.lr}

    def bank_dict(self):
        return {'bank': self.bank.detach().cpu().clone()}

    def load_bank(self, state):
        with torch.no_grad():
            self.bank.copy_(state['bank'].to(self.bank.device))


def install_kv_read(lm, memories):
    """Wrap _complete_kv on the given layers {layer_index: KVMemory}. Returns handles
    for removal. With memory.enabled False the wrapper is a pure pass-through, so the
    additive read is byte-identical to the un-wrapped path."""
    import moshi.modules.transformer as T
    handles = {}
    for L, mem in memories.items():
        attn = lm.transformer.layers[L].self_attn
        H = attn.num_heads
        D = attn.embed_dim // attn.num_heads
        assert H == mem.num_heads and D == mem.head_dim, (H, D, mem.num_heads, mem.head_dim)
        orig = attn._complete_kv

        def wrapped(k, v, _orig=orig, _mem=mem, _H=H, _D=D):
            res = _orig(k, v)
            if _mem.enabled:
                Km, Vm = _mem.read_kv()                       # [H, D]
                Km = Km.to(k.dtype).unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]
                Vm = Vm.to(k.dtype).unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]
                keys = torch.cat([Km, res.keys], dim=2)       # prepend one memory position
                values = torch.cat([Vm, res.values], dim=2)
                positions = torch.cat([torch.zeros(1, dtype=res.positions.dtype, device=res.positions.device),
                                       res.positions])
                return T.KVCacheResult(keys, values, positions)
            return res

        attn._complete_kv = wrapped
        handles[L] = (attn, orig)
    return handles


def remove_kv_read(handles):
    for attn, orig in handles.values():
        attn._complete_kv = orig
    handles.clear()
