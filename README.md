English | [简体中文](README_zh.md)

# EVA — Unofficial Implementation on the CAMS Framework: Decoupling Primitive with Experts — Dynamic Feature Alignment

> **Method provenance**: *Decoupling Primitive with Experts: Dynamic Feature Alignment for Compositional Zero-Shot Learning*, by Xiao Zhang, Haodong Jing, Yongqiang Ma, and Nanning Zheng (Xi'an Jiaotong University), **ICLR 2026** ([OpenReview](https://openreview.net/forum?id=hUtTGobe1r)).
>
> **About this repository**: the EVA paper has **no official open-source code**. This repository is an **unofficial PyTorch reimplementation written from scratch based on the paper's description**, built on top of the [CAMS](https://github.com/ybyangjing/CAMS) ([arXiv:2511.16378](https://arxiv.org/abs/2511.16378)) code framework. All credit for the method itself belongs to the EVA paper authors; this repository merely provides a runnable reference implementation.

## Method Overview (from the EVA paper)

Conventional CZSL methods learn a single prototype per primitive (state / object), yet the same primitive can vary drastically in semantics across compositions (e.g., *young dog* vs. *young city*), which a single prototype cannot cover. EVA addresses this with two mechanisms:

| Module | Description |
|--------|-------------|
| **Domain-Expert Adaption (MoE Adapter)** | A MoE Adapter connected **in parallel** to the FFN of every Transformer block in CLIP (`x = x + FFN(LN2(x)) + MoE(LN2(x))`): 1 shared expert + 8 routed experts (Router TopK=2), with LoRA-structured experts (r=64); applied to both the vision encoder (24 layers) and the text encoder (12 layers) |
| **Semantic Variant Alignment (SVA)** | Uses the independent CLS output of every expert in the last layer (without router weighting) to build a variant set `{v_i} = [B, 9, 768]`; via inter/intra-model affinity (I2T, weighted by `alpha`), dynamically selects the variants most relevant to the current input as the primitive visual features; **I2T is used for training only** (no GT labels at inference time, so variants are not used) |

Backbone: CLIP ViT-L/14 (fully frozen); only the newly added modules are trained (~39M / 467M, 8.5%).

**Training loss**: `L = L_c + λ1·(L_s + L_o) + λ2·(L^v_s + L^v_o)`, where `λ1 = 0.5` (T2I primitive loss, with learnable temperatures τ_s/τ_o), `λ2 = 0.1` (I2T variant loss), and `alpha = 0.5` (intra-model weight in the I2T affinity).

**Inference scoring** (Closed/Open World): `score(c=(s,o)|x) = p_c(c|x) + beta·(p_s(s|x) + p_o(o|x))`, where p_s/p_o are the partner-max of p_c, and `beta = 0.5`.

Implementation notes: no load-balancing loss is added, letting the MoE learn the semantic distribution naturally; the prompts follow the CSP style (the embedding of the 3 tokens in "a photo of" serves as a learnable prefix); the partner-max in `state_to_pairs` uses `-inf` rather than `-1`.

> Note: the details above are this author's implementation based on the paper's description and may differ from the official implementation (not released).

## Repository Contents

This repository **contains only the newly added/modified parts and none of the original upstream CAMS code** (upstream files such as `train.py`, `test.py`, `dataset.py`, `model/csp.py`, and `clip_modules/clip.py` are obtained by cloning CAMS). Modifications to upstream files are recorded as diffs in `patches_from_upstream/`.

```
EVA2/
├── clip_modules/
│   └── eva_gca.py                  # EVA Vision/Text Transformer + CLIP loader (implemented in this repo)
├── model/
│   ├── moe_adapter.py              # LoRAExpert / Router / MoEAdapter (implemented in this repo)
│   └── eva.py                      # Main EVA model: forward / loss / inference interfaces (implemented in this repo)
├── configs/
│   ├── mit-states-eva.yml          # epochs 20, lr 1e-4, bs 32
│   ├── utzappos-eva.yml            # epochs 20, lr 1e-4, wd 5e-4, bs 32
│   └── cgqa-eva.yml
├── predict_logits_eva.py           # EVA inference function
└── patches_from_upstream/          # Records of modifications to upstream files (for git apply)
    ├── model__configure_model.py.diff  # configure_model supports switching between EVA / CAMS
    ├── clip_modules__clip.py.diff
    ├── train.py.diff               # Hooks up predict_logits_eva
    ├── test.py.diff
    ├── utils.py.diff
    └── README.md.diff              # Historical record of upstream README → EVA description changes
```

Training hyperparameters (epochs=20 and lr=1e-4 for every dataset): MIT-States (weight_decay 1e-4, batch_size 32), UT-Zappos (5e-4), C-GQA (1e-4, 32).

## Reproduction Status

The numbers below were recorded as sanity checks during the development of this repository and are provided only as a reproduction reference:

**MIT-States Closed-World metrics reported in the paper (EVA paper, Table 2)**:

| Seen | Unseen | HM | AUC |
|------|--------|----|-----|
| 51.2 | 55.0 | 41.0 | 24.0 |

**Debug reference baselines of this implementation** (intermediate experiment records):

- MoE Adapter only (SVA removed) → AUC ≈ 23.0
- Prompt baseline only → AUC ≈ 20.2

Troubleshooting hint: if AUC does not improve, check whether the partner-max in `state_to_pairs` uses `-1` instead of `-inf`, and whether the text encoder's MoE parameters were added to the optimizer. This repository does not include full training logs; for the other datasets, please rely on your own reproduction outputs.

## Reproduction Steps

```bash
# 1. Clone the CAMS code framework (upstream code is not included in this repo)
git clone https://github.com/ybyangjing/CAMS.git EVA-work
cd EVA-work

# 2. Apply the modifications to upstream files
git apply /path/to/EVA2/patches_from_upstream/*.diff

# 3. Copy in the new files from this repository
#    clip_modules/eva_gca.py, model/moe_adapter.py, model/eva.py,
#    predict_logits_eva.py, configs/*-eva.yml
#    into the same relative paths under EVA-work/

# 4. Prepare the datasets (MIT-States / UT-Zappos / C-GQA) and the CLIP ViT-L/14 weights,
#    then update the clip_model path and dataset_path in the config

# 5. Train (evaluation on the test set runs automatically after training, printing best_seen / best_unseen / best_hm / AUC)
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mit-states-eva.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/utzappos-eva.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/cgqa-eva.yml

# 6. Standalone testing (set load_model in the config first; add --open_world for open-world evaluation)
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mit-states-eva.yml
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mit-states-eva.yml --open_world
```

The original CAMS configs (e.g., `configs/mit-states.yml`) still work after the patches are applied: `python train.py --config configs/mit-states.yml`. Environment dependencies and dataset formats are identical to upstream CAMS.

A second implementation of the same method can be found in the sibling repository [EVA3](https://github.com/Yyitan/EVA3): a reproduction on the Troika (CVPR 2024) framework, useful for cross-reference against this repository.

## Citation

The method and the engineering framework of this repository are respectively based on the following works. Please cite them if you use this repository:

```bibtex
@inproceedings{zhang2026eva,
  title={Decoupling Primitive with Experts: Dynamic Feature Alignment for Compositional Zero-Shot Learning},
  author={Zhang, Xiao and Jing, Haodong and Ma, Yongqiang and Zheng, Nanning},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

The implementation code written by the author of this repository is released under the MIT License (see LICENSE). The academic credit for the EVA method belongs to the paper's authors. The upstream CAMS code remains the property of its original authors, and the upstream repository declares no LICENSE.
