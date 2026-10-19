import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_modules.clip import tokenize
from clip_modules.eva_gca import load_eva_clip
from model.csp import CSPInterface


class EVA(nn.Module):
    """
    EVA: Decoupling Primitive with Experts for CZSL.
    Implements Domain-Expert Adaption (MoE Adapter) + Semantic Variant Alignment (SVA).
    """
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dset = dataset

        self.clip_model = load_eva_clip(
            path=args.clip_model,
            context_length=args.context_length,
            n_routed=getattr(args, 'n_routed', 8),
            k=getattr(args, 'k', 2),
            r=getattr(args, 'lora_r', 64),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_checkpoint=getattr(args, 'grad_checkpoint', False),
        )

        self.csp = CSPInterface(tokenize, self.clip_model, dataset, args)

        attr2idx = dataset.attr2idx
        obj2idx  = dataset.obj2idx
        self.train_pairs = torch.tensor(
            [(attr2idx[a], obj2idx[o]) for a, o in dataset.train_pairs])

        # Learnable temperature scalars for T2I alignment (Eq. 7/8)
        self.tau_s = nn.Parameter(torch.tensor(1.0))
        self.tau_o = nn.Parameter(torch.tensor(1.0))

        # Boolean masks for partner-max: [n_attrs, n_train_pairs] / [n_objs, n_train_pairs]
        n_a = len(dataset.attrs)
        n_o = len(dataset.objs)
        n_p = len(dataset.train_pairs)
        mask_S = torch.zeros(n_a, n_p, dtype=torch.bool)
        mask_O = torch.zeros(n_o, n_p, dtype=torch.bool)
        for idx, (a, o) in enumerate(dataset.train_pairs):
            mask_S[attr2idx[a], idx] = True
            mask_O[obj2idx[o],  idx] = True
        # persistent=False: rebuilt from dataset each time, should not be saved in checkpoint
        self.register_buffer('mask_S', mask_S, persistent=False)  # [n_a, n_p]
        self.register_buffer('mask_O', mask_O, persistent=False)  # [n_o, n_p]

        self.lambda_1 = getattr(args, 'lambda_1', 0.5)
        self.lambda_2 = getattr(args, 'lambda_2', 0.1)
        self.alpha    = getattr(args, 'alpha', 0.5)

    # ------------------------------------------------------------------ #
    #  Text feature computation                                            #
    # ------------------------------------------------------------------ #
    def _encode_text(self, token_ids, token_tensors):
        return self.clip_model.encode_text_tokens(token_ids, token_tensors)

    @property
    def _device(self):
        return next(self.clip_model.parameters()).device

    def get_text_features_train(self):
        """Compute normalised text features for all attrs, objs, and train pairs."""
        tok_c, tok_a, tok_o = self.csp.construct_token_tensors(self.train_pairs.to(self._device))
        T_A = F.normalize(self._encode_text(self.csp.token_ids[1], tok_a), p=2, dim=-1)
        T_O = F.normalize(self._encode_text(self.csp.token_ids[2], tok_o), p=2, dim=-1)
        T_C = F.normalize(self._encode_text(self.csp.token_ids[0], tok_c), p=2, dim=-1)
        return T_A, T_O, T_C  # [|S|,d], [|O|,d], [|C_tr|,d]

    # ------------------------------------------------------------------ #
    #  Training forward                                                    #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        imgs, attr_ids, obj_ids, pair_ids = batch[0], batch[1], batch[2], batch[3]
        B = imgs.shape[0]

        T_A, T_O, T_C = self.get_text_features_train()

        # Image features
        f_c, V = self.clip_model.encode_image(imgs)      # [B,d], [B,n_e+1,d]
        f_c    = F.normalize(f_c, p=2, dim=-1)
        V      = F.normalize(V,   p=2, dim=-1)

        logit_scale = self.clip_model.logit_scale.exp()

        # ── L_c: composition CE ──
        logits_c = f_c @ T_C.t() * logit_scale           # [B, |C_tr|]
        L_c      = F.cross_entropy(logits_c, pair_ids)

        # ── L_s, L_o: T2I primitive alignment (Eq. 7-8) ──
        p_c = F.softmax(logits_c, dim=-1)                 # [B, |C_tr|]

        # For each state s: max over partners (training pairs containing s)
        p_c_s  = p_c.unsqueeze(1).expand(-1, self.mask_S.shape[0], -1)  # [B,|S|,|C_tr|]
        p_s    = torch.where(
            self.mask_S.unsqueeze(0), p_c_s,
            torch.full_like(p_c_s, -1.0)
        ).max(dim=-1).values * self.tau_s                 # [B, |S|]

        p_c_o  = p_c.unsqueeze(1).expand(-1, self.mask_O.shape[0], -1)  # [B,|O|,|C_tr|]
        p_o    = torch.where(
            self.mask_O.unsqueeze(0), p_c_o,
            torch.full_like(p_c_o, -1.0)
        ).max(dim=-1).values * self.tau_o                 # [B, |O|]

        L_s = F.cross_entropy(p_s, attr_ids)
        L_o = F.cross_entropy(p_o, obj_ids)

        # ── L_v_s, L_v_o: I2T variant alignment (Eq. 9-14, training only) ──
        t_s_b = T_A[attr_ids]                             # [B, d]
        t_o_b = T_O[obj_ids]                              # [B, d]

        A_s = torch.einsum('bnd,bd->bn', V, t_s_b)       # [B, n_e+1]
        A_o = torch.einsum('bnd,bd->bn', V, t_o_b)
        A_v = torch.einsum('bnd,bd->bn', V, f_c)

        AS   = A_s + self.alpha * A_v
        AO   = A_o + self.alpha * A_v

        idx_s = AS.argmax(dim=-1)                          # [B]
        idx_o = AO.argmax(dim=-1)
        f_s   = V[torch.arange(B, device=V.device), idx_s]  # [B, d]
        f_o   = V[torch.arange(B, device=V.device), idx_o]

        L_v_s = F.cross_entropy(f_s @ T_A.t() * logit_scale, attr_ids)
        L_v_o = F.cross_entropy(f_o @ T_O.t() * logit_scale, obj_ids)

        loss = L_c + self.lambda_1 * (L_s + L_o) + self.lambda_2 * (L_v_s + L_v_o)
        return loss

    # ------------------------------------------------------------------ #
    #  Inference helper                                                    #
    # ------------------------------------------------------------------ #
    def encode_image_for_eval(self, imgs):
        """Return normalised image features (no variants needed at inference)."""
        f_c, _ = self.clip_model.encode_image(imgs)
        return F.normalize(f_c, p=2, dim=-1)

    def get_text_features_eval(self, pairs_tensor):
        """
        Compute text features for arbitrary pairs (used at test time).
        pairs_tensor: [N, 2] int tensor of (attr_idx, obj_idx)
        Returns: T_A [|S|,d], T_O [|O|,d], T_C [N,d]
        """
        tok_c, tok_a, tok_o = self.csp.construct_token_tensors(pairs_tensor)
        T_A = F.normalize(self._encode_text(self.csp.token_ids[1], tok_a), p=2, dim=-1)
        T_O = F.normalize(self._encode_text(self.csp.token_ids[2], tok_o), p=2, dim=-1)
        T_C = F.normalize(self._encode_text(self.csp.token_ids[0], tok_c), p=2, dim=-1)
        return T_A, T_O, T_C
