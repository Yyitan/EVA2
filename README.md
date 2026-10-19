# EVA 非官方实现（CAMS 框架）：Decoupling Primitive with Experts — Dynamic Feature Alignment

> **方法出处**：*Decoupling Primitive with Experts: Dynamic Feature Alignment for Compositional Zero-Shot Learning*，Xiao Zhang、Haodong Jing、Yongqiang Ma、Nanning Zheng（西安交通大学），**ICLR 2026**（[OpenReview](https://openreview.net/forum?id=hUtTGobe1r)）。
>
> **本仓库性质**：EVA 论文**未开源官方代码**，本仓库是本人按论文描述**从零实现的非官方 PyTorch 复现**，工程上搭建在 [CAMS](https://github.com/ybyangjing/CAMS)（[arXiv:2511.16378](https://arxiv.org/abs/2511.16378)）代码框架之上。方法的学术贡献归 EVA 论文作者所有，本仓库仅提供一份可运行的参考实现。

## 方法概述（来自 EVA 论文）

传统 CZSL 方法对每个 primitive（state / object）只学一个 prototype，但同一个 primitive 在不同 composition 下语义差异巨大（如 *young dog* vs *young city*），单一原型无法覆盖这种变化。EVA 用两组机制解决：

| 模块 | 描述 |
|------|------|
| **Domain-Expert Adaption（MoE Adapter）** | 在 CLIP 每个 Transformer Block 的 FFN 旁**并联** MoE Adapter（`x = x + FFN(LN2(x)) + MoE(LN2(x))`）：1 个 shared expert + 8 个 routed experts（Router TopK=2），expert 为 LoRA 结构（r=64）；同时作用于 vision encoder（24 层）与 text encoder（12 层） |
| **Semantic Variant Alignment（SVA）** | 用最后一层每个 expert 对 CLS 的独立输出（不经 router 加权）构造变体集 `{v_i} = [B, 9, 768]`，通过 inter/intra-model affinity（I2T，`alpha` 加权）动态选出与当前输入最相关的变体作为 primitive 视觉特征；**I2T 仅用于训练**（推理时无 GT label，不使用 variants） |

Backbone：CLIP ViT-L/14（全部冻结），仅训练新增模块（约 39M / 467M，8.5%）。

**训练损失**：`L = L_c + λ1·(L_s + L_o) + λ2·(L^v_s + L^v_o)`，其中 `λ1 = 0.5`（T2I primitive loss，可学习温度 τ_s/τ_o）、`λ2 = 0.1`（I2T variant loss）、`alpha = 0.5`（I2T affinity 中 intra-model 权重）。

**推理打分**（Closed/Open World）：`score(c=(s,o)|x) = p_c(c|x) + beta·(p_s(s|x) + p_o(o|x))`，其中 p_s/p_o 取 p_c 的 partner-max，`beta = 0.5`。

实现要点：不加 load-balancing loss，让 MoE 自然学习语义分布；prompt 沿用 CSP style（"a photo of" 3 token 的 embedding 作可学习前缀）；`state_to_pairs` 的 partner-max 用 `-inf` 而非 `-1`。

> 注：以上为本人依据论文描述的实现细节，与官方实现（未公开）可能存在差异。

## 仓库内容

本仓库**只包含新增/修改的部分，不含 CAMS 上游原始代码**（上游的 `train.py`、`test.py`、`dataset.py`、`model/csp.py`、`clip_modules/clip.py` 等由克隆 CAMS 获得）。对上游文件的修改以 diff 形式记录在 `patches_from_upstream/` 中。

```
EVA2/
├── clip_modules/
│   └── eva_gca.py                  # EVA Vision/Text Transformer + CLIP 加载器（本仓库实现）
├── model/
│   ├── moe_adapter.py              # LoRAExpert / Router / MoEAdapter（本仓库实现）
│   └── eva.py                      # EVA 主模型：前向/损失/推理接口（本仓库实现）
├── configs/
│   ├── mit-states-eva.yml          # epochs 20, lr 1e-4, bs 32
│   ├── utzappos-eva.yml            # epochs 20, lr 1e-4, wd 5e-4, bs 32
│   └── cgqa-eva.yml
├── predict_logits_eva.py           # EVA 推理函数
└── patches_from_upstream/          # 对上游文件的修改记录（git apply 用）
    ├── model__configure_model.py.diff  # configure_model 支持 EVA / CAMS 切换
    ├── clip_modules__clip.py.diff
    ├── train.py.diff               # 接入 predict_logits_eva
    ├── test.py.diff
    ├── utils.py.diff
    └── README.md.diff              # 上游 README → EVA 说明的历史修改记录
```

训练超参（各数据集 epochs=20、lr=1e-4）：MIT-States（weight_decay 1e-4，batch_size 32）、UT-Zappos（5e-4）、C-GQA（1e-4，32）。

## 复现情况

以下数字来自本仓库开发过程中记录的 sanity check，仅作复现对照参考：

**MIT-States Closed-World 论文报告指标（EVA 论文 Table 2）**：

| Seen | Unseen | HM | AUC |
|------|--------|----|-----|
| 51.2 | 55.0 | 41.0 | 24.0 |

**本实现的调试参考基线**（中间实验记录）：

- 仅加 MoE Adapter（去掉 SVA）→ AUC ≈ 23.0
- 仅 prompt baseline → AUC ≈ 20.2

排障提示：若 AUC 不升，检查 `state_to_pairs` 的 partner-max 是否用了 `-1` 而非 `-inf`，以及 text encoder 的 MoE 是否加入了 optimizer。本仓库未附带完整训练日志，其他数据集请以实际复现输出为准。

## 复现步骤

```bash
# 1. 克隆 CAMS 代码框架（本仓库不包含上游代码）
git clone https://github.com/ybyangjing/CAMS.git EVA-work
cd EVA-work

# 2. 应用对上游文件的修改
git apply /path/to/EVA2/patches_from_upstream/*.diff

# 3. 放入本仓库新增文件
#    clip_modules/eva_gca.py、model/moe_adapter.py、model/eva.py、
#    predict_logits_eva.py、configs/*-eva.yml
#    按相同相对路径复制进 EVA-work/

# 4. 准备数据集（MIT-States / UT-Zappos / C-GQA）与 CLIP ViT-L/14 权重，
#    修改 config 中的 clip_model 路径与 dataset_path

# 5. 训练（训练结束自动在 test set 评估，输出 best_seen / best_unseen / best_hm / AUC）
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mit-states-eva.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/utzappos-eva.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/cgqa-eva.yml

# 6. 单独测试（改 config 中 load_model 后运行；open-world 加 --open_world）
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mit-states-eva.yml
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mit-states-eva.yml --open_world
```

原 CAMS 配置（`configs/mit-states.yml` 等）在打完补丁后仍可正常使用：`python train.py --config configs/mit-states.yml`。环境依赖与数据集格式与上游 CAMS 一致。

同一方法的另一份实现见姊妹仓库 [EVA3](https://github.com/Yyitan/EVA3)：在 Troika（CVPR 2024）框架上的复现版本，可与本仓库交叉对照。

## 引用

本仓库的方法与工程框架分别基于以下工作，如使用请引用：

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

本仓库中本人撰写的实现代码以 MIT License 发布（见 LICENSE）。EVA 方法的学术贡献归论文作者所有；上游 CAMS 代码版权归原作者所有，且上游仓库未声明 LICENSE。
