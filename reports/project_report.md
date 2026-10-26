# 基于 ViT-B/16 的参数高效迁移学习方法比较

## 摘要

本项目围绕“小规模细粒度视觉任务中，如何高效迁移大型预训练模型”这一问题，
实现并比较了五种 ViT-B/16 迁移策略：Linear Probe、LoRA、原始分块 OFT、
BOFT 和 Full Fine-tuning。所有方法均使用相同的 ImageNet 预训练权重、
Oxford-IIIT Pet 数据划分、图像预处理、随机种子和 20-epoch 训练预算，最终在
官方测试集上评估。

项目没有直接封装完整的现成分类模型，而是使用 PyTorch 实现 ViT 的 Patch
Embedding、多头自注意力、MLP、Transformer Encoder、分类头以及 LoRA、OFT、
BOFT 适配层，再将 Hugging Face 发布的 `google/vit-base-patch16-224` 权重映射到
自定义网络。权重加载验证表明，自定义 backbone 与官方实现的输出逐元素一致。

实验结果显示，五种方法的测试准确率均超过 92.6%。BOFT 获得最高测试准确率
93.51%，LoRA 以 0.3755% 的可训练参数达到 93.38%，而原始分块 OFT 仅训练
240,421 个参数便达到 93.13%。Full Fine-tuning 更新全部 85,827,109 个参数，
测试准确率为 93.24%，没有超过 LoRA 或 BOFT。结果说明：对于该数据集，预训练
ViT 已具有很强的可迁移表征，参数高效方法能够以远低于全量微调的训练参数成本，
获得相当甚至更好的单次实验结果。

## 1. 研究背景与问题

大型预训练模型通常先在大规模通用数据上学习表征，再迁移到具体任务。传统的 Full
Fine-tuning 会更新网络的全部参数，表达能力充分，但训练显存、优化难度和任务检查点
成本较高。Parameter-Efficient Fine-Tuning（PEFT）则冻结大部分预训练参数，只学习
少量任务相关参数。

本项目关注三个具体问题：

1. ImageNet 预训练 ViT 在只训练分类头时，能够达到怎样的迁移性能？
2. LoRA、OFT 和 BOFT 能否用少量可训练参数超过 Linear Probe？
3. 低秩更新与正交更新在准确率、参数量和训练时间之间呈现怎样的权衡？

项目重点不是提出一个全新的迁移框架，而是建立一套结构透明、可验证、可扩展的实验
基线，为后续研究正交参数化、奇异值谱和网络稳定性提供代码与数据基础。

## 2. 数据集与任务

实验采用 Oxford-IIIT Pet 数据集，共包含 37 个猫狗品种，每类约 200 张图片。该任务
既具有图像分类的标准形式，又要求模型区分外观相近的品种，属于小规模细粒度分类。

官方 `trainval` 集按类别进行 8:2 分层划分；官方测试集只用于最终评估。所有实验固定
随机种子为 42。

| 数据部分 | 样本数 | 用途 |
|---|---:|---|
| 训练集 | 2,944 | 计算梯度和更新参数 |
| 验证集 | 736 | 每个 epoch 选择最佳检查点 |
| 官方测试集 | 3,669 | 对最佳检查点进行最终评估 |

训练图像经过 RGB 转换、随机缩放裁剪至 224×224、随机水平翻转、张量化与归一化。
验证集和测试集不使用随机增强，只进行确定性缩放和相同归一化。模型输入形状为

\[
X\in\mathbb R^{B\times3\times224\times224}.
\]

## 3. ViT-B/16 基础模型

ViT-B/16 将一张 224×224 图像划分为 16×16 的不重叠 patch，因此每张图像产生

\[
14\times14=196
\]

个 patch token。每个 patch 被线性投影到 768 维，再在序列开头加入一个可学习的
CLS token，并叠加可学习的位置编码。送入 Transformer Encoder 的张量形状为

\[
[B,197,768].
\]

主干包含 12 个 Transformer Encoder Block，每个 block 使用 Pre-LayerNorm 结构：

\[
X'=X+\operatorname{MSA}(\operatorname{LN}_1(X)),
\]

\[
X_{\mathrm{out}}=X'+\operatorname{MLP}(\operatorname{LN}_2(X')).
\]

多头注意力包含 Query、Key、Value 和输出投影，隐藏维度为 768，共 12 个 attention
head。MLP 的维度变化为

\[
768\rightarrow3072\rightarrow768,
\]

中间使用 GELU 激活。最终对 token 序列做 LayerNorm，取 CLS token 表征并输入
`Linear(768, 37)` 分类头，得到 37 个 logits。

## 4. 五种迁移策略

### 4.1 Linear Probe

Linear Probe 冻结整个 ViT backbone，只训练新分类头：

\[
z=\operatorname{ViT}_{\mathrm{frozen}}(x),\qquad
\hat y=W_{\mathrm{head}}z+b_{\mathrm{head}}.
\]

它只有 28,453 个可训练参数，是本项目最轻量的 baseline。该实验衡量预训练表征在不
调整 backbone 的情况下，能否直接支持新任务。

### 4.2 Full Fine-tuning

Full Fine-tuning 同时训练 backbone 与分类头。预训练 backbone 使用较小学习率
`2e-5`，随机初始化分类头使用 `1e-3`。该方法提供最大的优化自由度，同时需要保存完整
模型检查点，并对小数据集上的过拟合和优化波动更敏感。

### 4.3 LoRA

LoRA 冻结原线性权重 \(W\)，学习一个低秩增量：

\[
W'=W+\frac{\alpha}{r}BA,
\]

其中

\[
A\in\mathbb R^{r\times d_{\mathrm{in}}},\qquad
B\in\mathbb R^{d_{\mathrm{out}}\times r}.
\]

本项目取 \(r=8,\alpha=8\)，将 LoRA 注入全部 12 个 Encoder Block 的 Query 和
Value 投影。矩阵 \(B\) 初始化为零，因此训练开始时增量 \(BA=0\)，模型初始输出与
预训练模型一致。LoRA 参数为 294,912，加上分类头后共训练 323,365 个参数。

### 4.4 原始分块 OFT

OFT 不直接对权重添加任意增量，而是学习一个正交变换：

\[
W'=WR^T,\qquad R^TR=I.
\]

正交矩阵由反对称矩阵 \(Q\) 通过 Cayley 变换生成：

\[
R=(I-Q)(I+Q)^{-1},\qquad Q^T=-Q.
\]

若直接学习一个 768×768 的稠密正交矩阵，参数和计算成本仍然较高。本项目使用
block size 24，将 768 个特征维度划分为 32 个块，每个块学习独立的 24×24 正交
变换。实现仅存储反对称矩阵严格上三角的独立元素，因此每个块包含

\[
\frac{24\times23}{2}=276
\]

个自由度。OFT 注入全部 Query 和 Value 投影，共有 211,968 个 OFT 参数；连同分类
头共训练 240,421 个参数。

### 4.5 BOFT

单层分块 OFT 只能在各自 block 内混合特征。BOFT 将多个分块正交 factor 与固定
butterfly permutation 组合：

\[
R_{\mathrm{BOFT}}=B_6B_5\cdots B_1.
\]

本项目使用 block size 24 和 6 个 factor。每个 factor 的正交块参数彼此独立，不同
factor 之间使用固定置换重新组合坐标，使一个输出维度的有效信息覆盖范围逐步扩大：

\[
24\rightarrow48\rightarrow96\rightarrow192\rightarrow384\rightarrow768.
\]

因此 BOFT 仍由小型正交块构成，但六层后可以实现跨越完整 768 维特征空间的混合。
该配置包含 1,271,808 个 BOFT 参数，加上分类头后共训练 1,300,261 个参数。

## 5. 统一实验设置

除 Full Fine-tuning 因显存需求使用 batch size 16 外，其余方法均使用 batch size 32。
验证和测试 batch size 均为 64。所有方法训练 20 个 epoch，每轮结束后在验证集评估，
并保存验证准确率最高的检查点。训练完成后恢复该检查点，再评估官方测试集。

| 方法 | Backbone 学习率 | Adapter 学习率 | 分类头学习率 | Batch size |
|---|---:|---:|---:|---:|
| Linear Probe | 冻结 | — | `1e-3` | 32 |
| LoRA | 冻结 | `1e-4` | `1e-3` | 32 |
| 原始分块 OFT | 冻结 | `1e-4` | `1e-3` | 32 |
| BOFT | 冻结 | `1e-4` | `1e-3` | 32 |
| Full Fine-tuning | `2e-5` | — | `1e-3` | 16 |

共同设置如下：

| 项目 | 配置 |
|---|---|
| Epoch | 20 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Loss | CrossEntropyLoss |
| Random seed | 42 |
| 训练精度 | 支持时使用 BF16 自动混合精度 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER，16 GB |
| PyTorch | 2.5.1 + CUDA 12.4 |

## 6. 工程实现与验证

项目将共享组件和不同方法分离：`vit_core/` 负责 ViT 主干、数据集、预训练权重映射、
训练循环、评估和检查点；`experiments/` 下的各目录只保存方法特有的适配层、模型配置
和运行入口。这种结构避免为每个实验复制一整套 `net.py`、`train.py` 和 `main.py`。

预训练加载器将 198 个权重张量映射到自定义 ViT。使用相同随机输入比较自定义
backbone 与官方实现，得到：

```text
max_abs_difference:  0.0
mean_abs_difference: 0.0
```

项目测试进一步检查了：

- LoRA 零增量初始化与可训练参数范围；
- OFT/BOFT Cayley block 的正交性；
- 稀疏适配前向与显式构造 \(W'\) 的等价性；
- BOFT 多 factor 乘积的整体正交性；
- 预训练参数是否全部正确冻结或解冻；
- Linear Probe、Full Fine-tuning 和各 PEFT 方法的完整工作流。

OFT 训练时曾观察到 Windows CUDA 环境对大量重复小型 `torch.linalg.solve` 调用的
后端异常。最终实现将 24 个 Query/Value OFT 层的 Cayley 问题在每次 ViT 前向开始时
合并为一次可微分批量求解，再将结果分发给各层。该优化不改变公式、参数量、输出或
梯度，只减少底层求解器调用次数。优化后的 20-epoch 训练完整通过。

## 7. 实验结果

### 7.1 准确率与参数量

| 方法 | 最佳验证准确率 | 测试准确率 | 正确数 / 总数 | 模型总参数 | 可训练参数 | 可训练比例 |
|---|---:|---:|---:|---:|---:|---:|
| Linear Probe | 94.70% | 92.61% | 3,398 / 3,669 | 85,827,109 | 28,453 | 0.0332% |
| LoRA，Q+V，rank=8 | **95.52%** | 93.38% | 3,426 / 3,669 | 86,122,021 | 323,365 | 0.3755% |
| 原始 OFT，Q+V，b=24 | 95.38% | 93.13% | 3,417 / 3,669 | 86,039,077 | 240,421 | 0.2794% |
| BOFT，Q+V，b=24，m=6 | 94.84% | **93.51%** | **3,431 / 3,669** | 87,098,917 | 1,300,261 | 1.4929% |
| Full Fine-tuning | 95.38% | 93.24% | 3,421 / 3,669 | 85,827,109 | 85,827,109 | 100% |

### 7.2 实测时间

| 方法 | 20-epoch 训练时间 | 最终测试时间 |
|---|---:|---:|
| 原始分块 OFT | 217.27 秒（3 分 37 秒） | 12.97 秒 |
| BOFT | 414.02 秒（6 分 54 秒） | 15.09 秒 |
| Linear Probe | 早期运行未记录 | 早期运行未记录 |
| LoRA | 早期运行未记录 | 早期运行未记录 |
| Full Fine-tuning | 早期运行未记录 | 早期运行未记录 |

没有对缺失时间进行事后估算，以免将推测值与实测值混在一起。此外，Full Fine-tuning
使用 batch size 16，其他方法使用 batch size 32，因此即使补测时间，也需要注明这项
实验条件差异。

## 8. 结果分析

### 8.1 预训练表征本身已经很强

Linear Probe 只训练 28,453 个分类头参数，测试准确率仍达到 92.61%。这说明
ImageNet 预训练 ViT 提取的通用视觉特征已经能够较好地区分宠物品种。PEFT 方法面对的
不是一个弱 baseline，而是在强预训练表征上进一步调整决策边界和特征空间。

### 8.2 LoRA 具有最均衡的单次实验表现

LoRA 测试准确率为 93.38%，比 Linear Probe 高 0.77 个百分点，多正确分类 28 张
图片；同时只训练 323,365 个参数。它距离最高的 BOFT 仅差 0.13 个百分点，但可训练
参数只有 BOFT 的约 24.87%。在当前配置中，LoRA 提供了最均衡的准确率—参数量折中。

### 8.3 OFT 证明局部正交调整是有效的

原始分块 OFT 测试准确率为 93.13%，比 Linear Probe 高 0.52 个百分点，多正确分类
19 张图片。它只训练 240,421 个参数，比 rank-8 LoRA 少 25.65%。因此，即使每个
24 维 block 彼此独立，保持范数的局部旋转也能够有效调整预训练特征。

### 8.4 BOFT 提升了正交适配的表达能力

BOFT 测试准确率为 93.51%，比原始 OFT 高 0.38 个百分点，多正确分类 14 张图片。
这与 BOFT 的设计动机一致：固定置换使不同 block 的信息逐层交换，克服单层分块 OFT
只能局部旋转的限制。

但 BOFT 使用 1,300,261 个可训练参数，是原始 OFT 的约 5.41 倍；训练时间也是
OFT 的约 1.91 倍。因此当前 OFT—BOFT 对照同时改变了 factor 数量、参数容量和计算量，
不能被解释为严格的单变量消融实验。它说明增加 butterfly factors 有潜力提高性能，
但尚不能证明提升完全来自更好的结构而不是更多参数。

### 8.5 全量微调没有体现参数优势

Full Fine-tuning 测试准确率为 93.24%，比 Linear Probe 高 0.63 个百分点，却低于
LoRA 0.14 个百分点、低于 BOFT 0.27 个百分点。训练后期，训练准确率接近 100%，而
验证指标出现明显波动，表现出小数据集全量微调容易过拟合和优化不稳定的问题。

这不意味着 Full Fine-tuning 在一般情况下弱于 PEFT，只说明在当前数据规模、学习率、
batch size 和 20-epoch 设置下，更新全部参数没有转化成更好的测试结果。学习率调度、
warmup、正则化或更早停止可能改善其表现。

### 8.6 验证集与测试集排名并不完全一致

LoRA 获得最高验证准确率 95.52%，但 BOFT 获得最高测试准确率 93.51%。BOFT 的最佳
验证准确率反而只有 94.84%。当方法间差距只有零点几个百分点时，单次数据划分和随机
训练波动足以改变排名。因此应避免仅根据当前一次结果断言某种方法普遍优于其他方法。

## 9. 当前结论的边界

本项目已经形成完整、可运行的比较框架，但仍存在以下限制：

1. 每种方法目前只运行一个随机种子，尚未报告均值和标准差；
2. LoRA 只测试了 rank 8，没有完成 rank 消融；
3. 所有 adapter 都作用于 Query 和 Value，没有比较 Query-only、Value-only 或更多目标层；
4. OFT 与 BOFT 使用相同 block size，但参数量不匹配；
5. 仅使用一个数据集，不能说明结论能推广到其他视觉任务；
6. Linear Probe、LoRA、Full Fine-tuning 的早期运行没有记录墙钟时间；
7. 尚未测量严格的峰值显存和推理吞吐量。

因此，当前结果适合作为毕业论文的实验基础和初步观察，而不应被包装成统计稳定的普遍
结论。

## 10. 后续实验建议

按照研究价值和实施成本，建议依次完成：

1. 对五种方法使用 3 个随机种子重复实验，报告均值和标准差；
2. 测试 LoRA rank \(r=2,4,8,16\)，分析参数量与准确率关系；
3. 比较 Query-only、Value-only 和 Query+Value；
4. 设计与 LoRA 参数量接近的 BOFT 配置，进行参数匹配比较；
5. 改变 OFT block size，研究局部旋转范围与参数效率；
6. 统一补测训练时间、峰值显存、检查点大小和推理速度；
7. 使用学习率调度和 early stopping 改善 Full Fine-tuning baseline。

其中，第 2、4、5 项最适合进一步形成毕业论文中的消融实验。它们能够把当前的工程
比较推进为一个更明确的研究问题：在固定参数预算下，低秩加性更新与结构化正交旋转，
哪一种更有效？

## 11. 项目复现

下载数据集和预训练权重：

```powershell
cd D:\ViT-Transfer-Learning
E:\anaconda\python.exe -m scripts.download_pet_dataset
E:\anaconda\python.exe -m scripts.download_pretrained_model
```

运行五种实验：

```powershell
E:\anaconda\python.exe -m experiments.linear_probe.run_linear_probe
E:\anaconda\python.exe -m experiments.lora.run_lora
E:\anaconda\python.exe -m experiments.oft.run_oft
E:\anaconda\python.exe -m experiments.boft.run_boft
E:\anaconda\python.exe -m experiments.full_finetune.run_full_finetune
```

验证实现：

```powershell
E:\anaconda\python.exe -m tests.test_lora
E:\anaconda\python.exe -m tests.test_oft
E:\anaconda\python.exe -m tests.test_boft
E:\anaconda\python.exe -m tests.test_linear_probe
E:\anaconda\python.exe -m tests.test_full_finetune
E:\anaconda\python.exe -m tests.test_pretrained_loading
```

## 12. 总结

本项目完成了一个从 ViT 结构实现、预训练权重加载、数据处理、训练评估到多种迁移方法
比较的完整视觉迁移学习流程。实验表明：

- 预训练 ViT 在新任务上具有很强的直接迁移能力；
- LoRA、OFT、BOFT 均能以不足 1.5% 的可训练参数获得有竞争力的结果；
- BOFT 在当前单次实验中准确率最高；
- LoRA 给出了最均衡的准确率与参数量折中；
- 原始 OFT 展示了正交旋转在极少参数下的有效性；
- Full Fine-tuning 并未因更新全部参数而获得最佳结果。

更重要的是，项目已经建立了一套可继续扩展的实验框架。后续通过多随机种子、参数匹配
和 block/rank 消融，可以进一步回答低秩适配与正交适配之间更有研究价值的问题。

## 参考文献

1. Dosovitskiy, A. et al. [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR, 2021.
2. Hu, E. J. et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR, 2022.
3. Qiu, Z. et al. [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280). NeurIPS, 2023.
4. Liu, W. et al. [Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization](https://arxiv.org/abs/2311.06243). ICLR, 2024.
5. Parkhi, O. M. et al. [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/). CVPR, 2012.
