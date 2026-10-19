import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import LayerNorm, QuickGELU
from model.moe_adapter import MoEAdapter
from torch.utils.checkpoint import checkpoint as grad_ckpt


# ─────────────────────── Vision blocks ───────────────────────

class EVAResidualAttentionBlock(nn.Module):
    """CLIP vision block + parallel MoE Adapter.
    Last block also returns per-expert CLS variants for SVA.
    """
    def __init__(self, d_model, n_head, attn_mask=None,
                 n_routed=8, k=2, r=64, return_variants=False):
        super().__init__()
        self.attn            = nn.MultiheadAttention(d_model, n_head)
        self.ln_1            = LayerNorm(d_model)
        self.mlp             = nn.Sequential(OrderedDict([
            ('c_fc',  nn.Linear(d_model, d_model * 4)),
            ('gelu',  QuickGELU()),
            ('c_proj', nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2            = LayerNorm(d_model)
        self.attn_mask       = attn_mask
        self.moe             = MoEAdapter(d_model, n_routed, k, r)
        self.return_variants = return_variants

    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
               if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x):
        # x: [T, B, d]
        x      = x + self.attention(self.ln_1(x))
        normed = self.ln_2(x)                              # [T, B, d]
        nb     = normed.permute(1, 0, 2)                   # [B, T, d]
        moe    = self.moe(nb).permute(1, 0, 2)             # [T, B, d]
        x      = x + self.mlp(normed) + moe

        if self.return_variants:
            cls_n    = nb[:, 0:1, :]                       # [B, 1, d]
            variants = self.moe.get_cls_variants(cls_n)    # [B, n_e+1, d]
            return x, variants
        return x


class _VisTrf(nn.Module):
    """Container with .resblocks to match CLIP weight-key naming."""
    def __init__(self, blocks):
        super().__init__()
        self.resblocks = nn.ModuleList(blocks)


class EVAVisionTransformer(nn.Module):
    """CLIP ViT with MoE Adapter at every block.
    Weight keys mirror original CLIP so load_state_dict(strict=False) loads pretrained weights.
    Returns (f_c [B,d], V [B,n_e+1,d]).
    """
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim,
                 n_routed=8, k=2, r=64, use_checkpoint=False):
        super().__init__()
        self.output_dim     = output_dim
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        sc = width ** -0.5
        self.class_embedding      = nn.Parameter(sc * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            sc * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre  = LayerNorm(width)
        blocks = [
            EVAResidualAttentionBlock(
                width, heads, n_routed=n_routed, k=k, r=r,
                return_variants=(i == layers - 1))
            for i in range(layers)
        ]
        self.transformer = _VisTrf(blocks)
        self.ln_post = LayerNorm(width)
        self.proj    = nn.Parameter(sc * torch.randn(width, output_dim))

    def forward(self, x):
        x   = self.conv1(x)
        x   = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + \
              torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.positional_embedding.to(x.dtype)
        x   = self.ln_pre(x).permute(1, 0, 2)             # [T, B, w]

        blks     = self.transformer.resblocks
        variants = None
        for i, blk in enumerate(blks):
            if i == len(blks) - 1:
                # Last block: always normal forward (returns variants tuple)
                x, variants = blk(x)
            elif self.use_checkpoint and torch.is_grad_enabled():
                x = grad_ckpt(blk, x)
            else:
                x = blk(x)

        x   = x.permute(1, 0, 2)                          # [B, T, w]
        f_c = self.ln_post(x[:, 0, :]) @ self.proj        # [B, output_dim]
        V   = (self.ln_post(variants) @ self.proj) if variants is not None else None
        return f_c, V                                      # V: [B, n_e+1, output_dim]


# ─────────────────────── Text blocks ─────────────────────────

class EVATextResidualAttentionBlock(nn.Module):
    """CLIP text block + parallel MoE Adapter."""
    def __init__(self, d_model, n_head, attn_mask=None, n_routed=8, k=2, r=64):
        super().__init__()
        self.attn      = nn.MultiheadAttention(d_model, n_head)
        self.ln_1      = LayerNorm(d_model)
        self.mlp       = nn.Sequential(OrderedDict([
            ('c_fc',  nn.Linear(d_model, d_model * 4)),
            ('gelu',  QuickGELU()),
            ('c_proj', nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2      = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.moe       = MoEAdapter(d_model, n_routed, k, r)

    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
               if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x):
        # x: [T, B, d]
        x      = x + self.attention(self.ln_1(x))
        normed = self.ln_2(x)
        nb     = normed.permute(1, 0, 2)
        moe    = self.moe(nb).permute(1, 0, 2)
        x      = x + self.mlp(normed) + moe
        return x


class _TxtTrf(nn.Module):
    """Container with .resblocks to match CLIP weight-key naming."""
    def __init__(self, blocks, use_checkpoint=False):
        super().__init__()
        self.resblocks      = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for blk in self.resblocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                x = grad_ckpt(blk, x)
            else:
                x = blk(x)
        return x


# ─────────────────── Full EVA CLIP model + loader ────────────────────

class EVACLIPModel(nn.Module):
    """Minimal CLIP with EVA MoE blocks. Key names match original CLIP for weight loading."""
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size, transformer_width,
                 transformer_heads, transformer_layers, n_routed=8, k=2, r=64,
                 use_checkpoint=False):
        super().__init__()
        self.context_length = context_length
        self.visual = EVAVisionTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size,
            width=vision_width, layers=vision_layers, heads=vision_width // 64,
            output_dim=embed_dim, n_routed=n_routed, k=k, r=r,
            use_checkpoint=use_checkpoint)

        mask = self._causal_mask(context_length)
        self.transformer = _TxtTrf([
            EVATextResidualAttentionBlock(
                transformer_width, transformer_heads, mask, n_routed, k, r)
            for _ in range(transformer_layers)
        ], use_checkpoint=use_checkpoint)
        self.token_embedding      = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final             = LayerNorm(transformer_width)
        self.text_projection      = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale          = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @staticmethod
    def _causal_mask(ctx):
        m = torch.empty(ctx, ctx).fill_(float('-inf'))
        m.triu_(1)
        return m

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, x):
        return self.visual(x)          # (f_c, V)

    def encode_text_tokens(self, token_ids, token_tensors):
        """
        Encode text from pre-built CSP token embeddings.
        token_ids:     [context_length]     - to find EOS position
        token_tensors: [N, context_length, d]
        Returns:       [N, embed_dim]
        """
        x = token_tensors.to(self.text_projection.dtype)
        x = x + self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)        # [T, N, d]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)        # [N, T, d]
        x = self.ln_final(x).to(self.text_projection.dtype)
        return x[:, int(token_ids.argmax()), :] @ self.text_projection


def load_eva_clip(path, context_length=77, n_routed=8, k=2, r=64, device='cuda',
                  use_checkpoint=False):
    """Load CLIP ViT-L/14 weights into EVACLIPModel. MoE weights are randomly initialised."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if hasattr(sd, 'state_dict'):
        sd = sd.state_dict()

    vw  = sd['visual.conv1.weight'].shape[0]
    vl  = len([kk for kk in sd if kk.startswith('visual.') and kk.endswith('.attn.in_proj_weight')])
    vps = sd['visual.conv1.weight'].shape[-1]
    gs  = round((sd['visual.positional_embedding'].shape[0] - 1) ** 0.5)
    res = vps * gs
    ed  = sd['text_projection'].shape[1]
    vs  = sd['token_embedding.weight'].shape[0]
    tw  = sd['ln_final.weight'].shape[0]
    th  = tw // 64
    tl  = len(set(kk.split('.')[2] for kk in sd if kk.startswith('transformer.resblocks')))

    if context_length != 77:
        pe = sd['positional_embedding']
        if context_length > 77:
            sd['positional_embedding'] = torch.cat([pe, torch.zeros(context_length - 77, tw)], 0)
        else:
            sd['positional_embedding'] = pe[:context_length]

    model = EVACLIPModel(ed, res, vl, vw, vps, context_length, vs, tw, th, tl,
                         n_routed, k, r, use_checkpoint=use_checkpoint)
    inc   = model.load_state_dict(sd, strict=False)
    print(f'[EVA CLIP] missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}')
    if use_checkpoint:
        print('[EVA CLIP] Gradient checkpointing ON (trades compute for memory)')
    return model.to(device)
