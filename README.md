# EVA2：EVA — Decoupling Primitive with Experts（MoE Adapter + Semantic Variant Alignment 的动态特征对齐 CZSL 方法）

> 基于 [CAMS](https://github.com/ybyangjing/CAMS)（[arXiv:2511.16378](https://arxiv.org/abs/2511.16378)）的改进研究 · 个人研究项目
>
> 论文：*Decoupling Primitive with Experts: Dynamic Feature Alignment for Compositional Zero-Shot Learning*（ICLR 2026 投稿）

## 研究动机

传统 CZSL 方法对每个 primitive（state / object）只学一个 prototype，但同一个 primitive 在不同 composition 下语义差异巨大（如 *young dog* vs *young city*），单一原型无法覆盖这种变化。EVA 用两组机制解决：(1) 在 CLIP 每层内部并联 MoE Adapter，让不同 expert 承接同一 primitive 在不同上下文中的变化；(2) 用最后一层各 expert 的独立输出构造语义变体集，推理/训练时动态选出与当前输入最相关的变体作为 primitive 视觉特征，实现 primitive 级的动态特征对齐。

## 方法

Backbone：CLIP ViT-L/14（全部冻结），仅训练新增模块（约 39M / 467M，8.5%）。

| 模块 | 描述 |
|------|------|
| **Domain-Expert Adaption（MoE Adapter）** | 在每个 Transformer Block 的 FFN 旁**并联** MoE Adapter（`x = x + FFN(LN2(x)) + MoE(LN2(x))`）：1 个 shared expert + 8 个 routed experts（Router TopK=2），expert 为 LoRA 结构（r=64）；同时作用于 vision encoder（24 层）与 text encoder（12 层） |
| **Semantic Variant Alignment（SVA）** | 用最后一层每个 expert 对 CLS 的独立输出（不经 router 加权）构造变体集 `{v_i} = [B, 9, 768]`，通过 inter/intra-model affinity（I2T，`alpha` 加权）动态选出与当前输入最相关的变体作为 primitive 视觉特征；**I2T 仅用于训练**（推理时无 GT label，不使用 variants） |

**训练损失**：`L = L_c + λ1·(L_s + L_o) + λ2·(L^v_s + L^v_o)`，其中 `λ1 = 0.5`（T2I primitive loss，可学习温度 τ_s/τ_o）、`λ2 = 0.1`（I2T variant loss）、`alpha = 0.5`（I2T affinity 中 intra-model 权重）。

**推理打分**（Closed/Open World）：`score(c=(s,o)|x) = p_c(c|x) + beta·(p_s(s|x) + p_o(o|x))`，其中 p_s/p_o 取 p_c 的 partner-max，`beta = 0.5`。

实现要点：不加 load-balancing loss，让 MoE 自然学习语义分布；prompt 沿用 CSP style（"a photo of" 3 token 的 embedding 作可学习前缀）；`state_to_pairs` 的 partner-max 用 `-inf` 而非 `-1`。

| 文件 | 说明 |
|------|------|
| `model/moe_adapter.py` | 新增：`LoRAExpert` / `Router` / `MoEAdapter` |
| `model/eva.py` | 新增：EVA 主模型（前向 / 损失 / T2I 对齐 / 掩码构造） |
| `clip_modules/eva_gca.py` | 新增：EVA 版 Vision/Text Transformer + CLIP 加载器 `load_eva_clip` |
| `predict_logits_eva.py` | 新增：EVA 推理函数（train.py 自动调用） |
| `configs/*-eva.yml` | MIT-States / UT-Zappos / C-GQA 的 EVA 配置 |

## 仓库内容

本仓库**只包含新增/修改的部分，不含 CAMS 上游原始代码**（上游的 `train.py`、`test.py`、`dataset.py`、`model/csp.py`、`clip_modules/clip.py` 等由克隆 CAMS 获得）。对上游文件的修改以 diff 形式记录在 `patches_from_upstream/` 中。

```
EVA2/
├── clip_modules/
│   └── eva_gca.py                  # EVA Vision/Text Transformer + CLIP 加载器（核心新增）
├── model/
│   ├── moe_adapter.py              # LoRAExpert / Router / MoEAdapter（核心新增）
│   └── eva.py                      # EVA 主模型：前向/损失/推理接口（核心新增）
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
    └── README.md.diff              # 上游 README → EVA README 的历史修改记录
```

训练超参（各数据集 epochs=20、lr=1e-4）：MIT-States（weight_decay 1e-4，batch_size 32）、UT-Zappos（5e-4）、C-GQA（1e-4，32）。

## 实验结果

以下数字来自本仓库原 README 中记录的 sanity check（`patches_from_upstream/README.md.diff`），仅作复现对照参考：

**MIT-States Closed-World 预期指标（EVA 论文 Table 2）**：

| Seen | Unseen | HM | AUC |
|------|--------|----|-----|
| 51.2 | 55.0 | 41.0 | 24.0 |

**调试参考基线**（原 README 记录的中间实验）：

- 仅加 MoE Adapter（去掉 SVA）→ AUC ≈ 23.0
- 仅 prompt baseline → AUC ≈ 20.2

原 README 同时给出排障提示：若 AUC 不升，检查 `state_to_pairs` 的 partner-max 是否用了 `-1` 而非 `-inf`，以及 text encoder 的 MoE 是否加入了 optimizer。本仓库未附带完整训练日志，其他数据集请以实际复现输出为准。

## 复现步骤

```bash
# 1. 克隆上游 CAMS（本仓库不包含上游代码）
git clone https://github.com/ybyangjing/CAMS.git EVA2-work
cd EVA2-work

# 2. 应用对上游文件的修改
git apply /path/to/EVA2/patches_from_upstream/*.diff

# 3. 放入本仓库新增文件
#    clip_modules/eva_gca.py、model/moe_adapter.py、model/eva.py、
#    predict_logits_eva.py、configs/*-eva.yml
#    按相同相对路径复制进 EVA2-work/

# 4. 准备数据集（MIT-States / UT-Zappos / C-GQA）与 CLIP ViT-L/14 权重，
#    修改 config 中的 clip_model 路径（默认 ./lyt/czsl/CLIP-main/ViT-L-14.pt）
#    与 dataset_path（各 config 内 train.dataset_path 或 --dataset_path）

# 5. 训练（训练结束自动在 test set 评估，输出 best_seen / best_unseen / best_hm / AUC）
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mit-states-eva.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/utzappos-eva.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/cgqa-eva.yml

# 6. 单独测试（改 config 中 load_model 后运行；open-world 加 --open_world）
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mit-states-eva.yml
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mit-states-eva.yml --open_world
```

原 CAMS 配置（`configs/mit-states.yml` 等）在打完补丁后仍可正常使用：`python train.py --config configs/mit-states.yml`。环境依赖与数据集格式与上游 CAMS 一致。

## 引用

本仓库基于以下工作，如使用请同时引用：

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

本项目以 MIT License 发布（仅覆盖本仓库新增的文件；上游 CAMS 代码版权归原作者所有，且上游仓库未声明 LICENSE）。
