"""
Drop-in replacement for predict_logits when using the EVA model.
Paste / import this into test.py alongside predict_logits.
"""
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_logits_eva(model, dataset, args):
    """
    EVA inference (Eq. 16):
        score(c=(s,o)|x) = p_c(c|x) + beta * (p_s(s|x) + p_o(o|x))

    p_c is softmax over ALL test compositions.
    p_s / p_o are partner-max of p_c over test compositions.
    """
    model.eval()
    beta = getattr(args, 'beta', 0.5)

    attr2idx = dataset.attr2idx
    obj2idx  = dataset.obj2idx

    # ── Build text features for ALL test pairs ──────────────────────
    all_pairs   = dataset.pairs          # list of (attr, obj)
    pairs_tensor = torch.tensor(
        [(attr2idx[a], obj2idx[o]) for a, o in all_pairs], dtype=torch.long)

    n_pairs      = len(all_pairs)
    n_attrs      = len(dataset.attrs)
    n_objs       = len(dataset.objs)

    # Build partner-max masks for test time
    # mask_S_test[s, p] = True iff test pair p contains attr s
    mask_S_test = torch.zeros(n_attrs, n_pairs, dtype=torch.bool)
    mask_O_test = torch.zeros(n_objs,  n_pairs, dtype=torch.bool)
    for idx, (a, o) in enumerate(all_pairs):
        mask_S_test[attr2idx[a], idx] = True
        mask_O_test[obj2idx[o],  idx] = True

    # Encode text in batches (large for open-world)
    batch_text = getattr(args, 'eval_batch_size', 64)
    T_C_list   = []
    with torch.no_grad():
        for start in range(0, n_pairs, batch_text):
            end   = min(start + batch_text, n_pairs)
            chunk = pairs_tensor[start:end].cuda()
            tok_c, tok_a, tok_o = model.csp.construct_token_tensors(chunk)
            tc = F.normalize(
                model.clip_model.encode_text_tokens(model.csp.token_ids[0], tok_c),
                p=2, dim=-1)
            T_C_list.append(tc.cpu())
            if start == 0:
                # T_A, T_O are always full (construct_token_tensors covers all)
                T_A = F.normalize(
                    model.clip_model.encode_text_tokens(model.csp.token_ids[1], tok_a),
                    p=2, dim=-1).cpu()
                T_O = F.normalize(
                    model.clip_model.encode_text_tokens(model.csp.token_ids[2], tok_o),
                    p=2, dim=-1).cpu()
        T_C = torch.cat(T_C_list, dim=0)  # [n_pairs, d]

    # ── Iterate over images ──────────────────────────────────────────
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    logit_scale = model.clip_model.logit_scale.exp().item()

    all_logits   = []
    all_attr_gt  = []
    all_obj_gt   = []
    all_pair_gt  = []

    T_C = T_C.to(device)
    T_A = T_A.to(device)
    T_O = T_O.to(device)
    mask_S_test = mask_S_test.to(device)
    mask_O_test = mask_O_test.to(device)

    with torch.no_grad():
        for data in tqdm(dataloader, desc='EVA Eval'):
            data = [d.to(device) if not isinstance(d, tuple) else d for d in data]
            imgs = data[0]

            f_c, _ = model.clip_model.encode_image(imgs)
            f_c     = F.normalize(f_c, p=2, dim=-1)         # [B, d]

            logits_c   = f_c @ T_C.t() * logit_scale        # [B, n_pairs]
            p_c        = F.softmax(logits_c, dim=-1)         # [B, n_pairs]

            # Partner-max for each state / object over test pairs
            p_c_s = p_c.unsqueeze(1).expand(-1, n_attrs, -1) # [B, n_a, n_p]
            p_s   = torch.where(
                mask_S_test.unsqueeze(0), p_c_s,
                torch.full_like(p_c_s, -1.0)
            ).max(dim=-1).values                              # [B, n_a]

            p_c_o = p_c.unsqueeze(1).expand(-1, n_objs, -1)  # [B, n_o, n_p]
            p_o   = torch.where(
                mask_O_test.unsqueeze(0), p_c_o,
                torch.full_like(p_c_o, -1.0)
            ).max(dim=-1).values                              # [B, n_o]

            # score(c=(s,o)) = p_c[c] + beta*(p_s[s] + p_o[o])
            # pairs2attr_idx / pairs2obj_idx broadcast attr/obj scores back to pair dimension
            attr_scores = p_s[:, dataset.pairs2attr_idx.to(device)]  # [B, n_pairs]
            obj_scores  = p_o[:, dataset.pairs2obj_idx.to(device)]   # [B, n_pairs]
            score       = p_c + beta * (attr_scores + obj_scores)    # [B, n_pairs]

            all_logits.append(score.cpu())
            all_attr_gt.append(data[1].cpu())
            all_obj_gt.append(data[2].cpu())
            all_pair_gt.append(data[3].cpu())

    return (torch.cat(all_logits),
            torch.cat(all_attr_gt),
            torch.cat(all_obj_gt),
            torch.cat(all_pair_gt))
