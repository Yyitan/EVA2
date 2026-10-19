import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    def __init__(self, d_model: int, r: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class Router(nn.Module):
    def __init__(self, d_model: int, n_routed: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.gate(x)


class MoEAdapter(nn.Module):
    """Parallel MoE Adapter: 1 shared expert + n_routed routed (TopK active). Input: [B,T,d]."""
    def __init__(self, d_model: int, n_routed: int = 8, k: int = 2, r: int = 64):
        super().__init__()
        self.n_routed = n_routed
        self.k        = k
        self.shared   = LoRAExpert(d_model, r)
        self.experts  = nn.ModuleList([LoRAExpert(d_model, r) for _ in range(n_routed)])
        self.router   = Router(d_model, n_routed)

    def forward(self, x):
        # x: [B, T, d]
        logits = self.router(x)                             # [B, T, n_routed]
        topk_val, topk_idx = logits.topk(self.k, dim=-1)   # [B, T, k]
        masked = torch.full_like(logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_val)
        gates = F.softmax(masked, dim=-1)                   # 0 at non-topk positions

        shared_out = self.shared(x)
        routed_out = torch.zeros_like(x)
        for i, E in enumerate(self.experts):
            g_i = gates[..., i:i+1]                         # [B, T, 1]
            routed_out = routed_out + g_i * E(x)

        return shared_out + routed_out

    def get_cls_variants(self, cls_normed):
        """Each expert independently on CLS (no router weighting). Returns [B, n_routed+1, d]."""
        outs = [self.shared(cls_normed)]
        for E in self.experts:
            outs.append(E(cls_normed))
        return torch.cat(outs, dim=1)
