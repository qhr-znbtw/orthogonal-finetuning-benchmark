# 基于 LoRA 的 ViT-B/16 宠物品种分类实验报告

## 摘要

本项目研究如何利用低秩适配（Low-Rank Adaptation，LoRA）将 ImageNet 预训练的 Vision Transformer 迁移到小规模细粒度图像分类任务。实验采用 Oxford-IIIT Pet 数据集，以 `google/vit-base-patch16-224` 为预训练模型，在全部 12 个 Transformer Encoder Block 的 Query 和 Value 投影中加入 LoRA 分支，同时冻结原始 ViT 参数，仅训练 LoRA 参数与新构造的 37 类分类头。

在单张 NVIDIA GeForce RTX 4070 Ti SUPER 上训练 20 个 epoch 后，模型在第 9 个 epoch 达到最高验证准确率 95.52%。恢复该最佳检查点后，模型在官方测试集上取得 93.38% 的 Top-1 准确率，正确分类 3426/3669 张图片。完整模型包含 86,122,021 个参数，其中仅 323,365 个参数参与训练，占总参数量约 0.375%；最佳任务检查点大小约 1.25 MB。

当前结果证明了所实现的 ViT-LoRA 迁移流程能够以较少的可训练参数完成有效的细粒度宠物分类。在相同协议下，Linear Probe baseline 的测试准确率为 92.61%，LoRA 在本次单随机种子实验中高出 0.77 个百分点；但不同 LoRA rank 的消融实验以及多随机种子重复实验尚未完成，因此暂时不能据此断言这一优势具有统计稳定性。

## 1. 研究任务

本项目的目标是完成一个结构透明、代码简洁且可复现的预训练模型迁移实验：

1. 使用 ImageNet 预训练的 ViT-B/16 作为视觉主干；
2. 将原有 ImageNet 分类头替换为 Oxford-IIIT Pet 的 37 类分类头；
3. 冻结预训练模型原始参数；
4. 在每个 Transformer Encoder Block 的 Query 和 Value 投影中加入 LoRA；
5. 仅优化 LoRA 参数和新分类头；
6. 使用独立验证集选择最佳模型；
7. 在官方测试集上报告最终分类准确率。

当前实验回答的直接问题是：

> 在冻结 ViT-B/16 原始参数的条件下，仅训练 Query/Value LoRA 与分类头，能否在 Oxford-IIIT Pet 上获得有效的分类结果？

当前实验尚不能回答 LoRA 是否优于只训练分类头、全参数微调或其他参数高效微调方法。这些问题需要后续对照实验和消融实验。

## 2. 数据集与划分

实验采用 [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) 数据集。该数据集包含 37 个猫狗品种，每类约 200 张图片，图片在尺度、姿态和光照方面存在较大变化。

本项目保留官方测试集不参与训练或调参，并将官方 `trainval` 集按类别进行 8:2 分层划分。固定随机种子为 42。

| 数据部分 | 样本数 | 用途 |
|---|---:|---|
| 训练集 | 2944 | 计算梯度并更新 LoRA 与分类头 |
| 验证集 | 736 | 每个 epoch 评估并选择最佳检查点 |
| 官方测试集 | 3669 | 训练结束后进行一次最终评估 |

训练集和验证集均保留完整的 37 个类别。训练集每类包含 74～80 张图片，验证集每类包含 19～20 张图片。

### 2.1 图像预处理

训练集预处理包括：

1. 将图片转换为 RGB 三通道；
2. 随机裁剪并缩放至 224×224，裁剪比例范围为 0.8～1.0；
3. 随机水平翻转；
4. 转换为 PyTorch 张量；
5. 使用均值 `(0.5, 0.5, 0.5)` 和标准差 `(0.5, 0.5, 0.5)` 归一化。

验证集和测试集不使用随机增强，只进行固定尺寸缩放、张量化和相同的归一化。模型最终接收的输入形状为：

\[
X\in\mathbb{R}^{B\times3\times224\times224}.
\]

## 3. 模型方法

### 3.1 ViT-B/16 主干

本项目采用 `google/vit-base-patch16-224`。Vision Transformer 的基本思想来自论文 [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)：将图像划分为 patch，把 patch 转换为 token 序列，再使用 Transformer Encoder 建模全局关系。

对于一张 224×224 的 RGB 图片，ViT-B/16 使用 16×16 的不重叠 patch，因此产生：

\[
14\times14=196
\]

个 patch token。每个 token 的隐藏维度为 768。加入一个可学习的 CLS token 和位置编码后，送入 Encoder 的张量形状为：

\[
[B,197,768].
\]

ViT-B/16 包含 12 个 Transformer Encoder Block。每个 Block 由多头自注意力和 MLP 两部分组成：

\[
X'=X+\operatorname{MSA}(\operatorname{LN}_1(X)),
\]

\[
X_{\mathrm{out}}=X'+\operatorname{MLP}(\operatorname{LN}_2(X')).
\]

其中 MLP 的维度变化为：

\[
768\rightarrow3072\rightarrow768.
\]

最后对全部 token 进行 LayerNorm，取第一个 CLS token 的 768 维表示，输入新的线性分类头，输出 37 个类别 logits。

### 3.2 LoRA 设计

LoRA 的基本思想来自论文 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。对于原线性映射：

\[
y=Wx,
\]

冻结原权重 \(W\)，并加入低秩更新：

\[
y=Wx+\frac{\alpha}{r}BAx,
\]

其中：

\[
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
\]

本实验的 LoRA 配置如下：

| 配置项 | 取值 |
|---|---:|
| LoRA rank \(r\) | 8 |
| LoRA alpha \(\alpha\) | 8 |
| 缩放系数 \(\alpha/r\) | 1 |
| 目标模块 | 每层 Attention 的 Query、Value |
| Encoder Block 数 | 12 |
| LoRA dropout | 0 |
| LoRA bias | 无 |

矩阵 \(A\) 使用 Kaiming 均匀初始化，矩阵 \(B\) 初始化为全零。因此训练开始时：

\[
BA=0,
\]

加入 LoRA 后的模型初始输出与原预训练主干保持一致。

每个 Query 或 Value LoRA 分支包含：

\[
768\times8+8\times768=12,288
\]

个参数。12 层中的 Query 和 Value 共包含：

\[
12,288\times2\times12=294,912
\]

个 LoRA 参数。37 类分类头包含：

\[
768\times37+37=28,453
\]

个参数。因此总可训练参数量为：

\[
294,912+28,453=323,365.
\]

| 参数统计 | 数量 |
|---|---:|
| 完整模型参数 | 86,122,021 |
| LoRA 参数 | 294,912 |
| 分类头参数 | 28,453 |
| 总可训练参数 | 323,365 |
| 可训练参数比例 | 0.375% |

## 4. 预训练权重加载与正确性验证

项目下载了 `google/vit-base-patch16-224` 的 `model.safetensors`、模型配置和预处理配置。自定义网络从权重文件中加载了 198 个预训练张量，包括：

- Patch Embedding；
- CLS token；
- 位置编码；
- 12 层 Encoder 中的 LayerNorm、Query、Key、Value、Attention 输出投影和 MLP；
- Encoder 后的最终 LayerNorm。

原 ImageNet 1000 类分类头未加载；LoRA 参数和新的 37 类分类头保留自定义初始化。

为验证自定义实现和权重映射的正确性，使用相同随机输入比较自定义 backbone 与 Hugging Face 官方 ViT 实现的最终 token 输出，结果为：

```text
max_abs_difference:  0.0
mean_abs_difference: 0.0
```

这表明在 LoRA 增量为零时，自定义网络的前向计算与官方实现逐元素一致。

## 5. 训练配置

### 5.1 硬件与软件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| GPU 显存 | 16 GB |
| PyTorch | 2.5.1 + CUDA 12.4 |
| 训练精度 | BF16 自动混合精度 |
| Python 环境 | `E:\anaconda\python.exe` |

正式训练过程中观察到 GPU 利用率约 99%，显存占用约 5.0 GB。该数值为训练过程中的监控快照，不作为严格峰值显存测量结果。

### 5.2 优化设置

| 超参数 | 取值 |
|---|---:|
| Epoch | 20 |
| 训练 batch size | 32 |
| 验证/测试 batch size | 64 |
| 每个 epoch 的训练更新次数 | 92 |
| 总参数更新次数 | 1840 |
| 优化器 | AdamW |
| LoRA 学习率 | \(1\times10^{-4}\) |
| 分类头学习率 | \(1\times10^{-3}\) |
| Weight decay | 0.01 |
| 损失函数 | CrossEntropyLoss |
| 随机种子 | 42 |
| 学习率调度器 | 无 |
| Label smoothing | 无 |
| 梯度累积 | 无 |
| 梯度裁剪 | 无 |

LoRA 和分类头采用不同学习率：随机初始化的分类头需要更快地适应新类别，而 LoRA 只需对预训练表征进行较小调整。

## 6. 实验流程

每个 epoch 执行以下过程：

1. 在训练集上进行前向传播、交叉熵计算、反向传播和 AdamW 更新；
2. 切换至评估模式，在验证集上进行无梯度评估；
3. 若当前验证准确率高于历史最佳值，则保存当前 LoRA 与分类头参数；
4. 继续下一个 epoch。

训练完成后，重新加载验证准确率最高的检查点，再在官方测试集上执行一次最终评估。测试集未参与训练、超参数调整或最佳 epoch 选择。

保存的检查点仅包含 48 个 LoRA 权重张量和分类头的 weight、bias，共 50 个张量，不重复保存冻结的 ViT 主干。

## 7. 实验结果

### 7.1 完整训练记录

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.8381 | 82.07% | 0.2821 | 91.85% |
| 2 | 0.1802 | 95.89% | 0.2151 | 93.48% |
| 3 | 0.1159 | 96.88% | 0.1997 | 93.89% |
| 4 | 0.0787 | 97.93% | 0.1732 | 95.11% |
| 5 | 0.0563 | 98.64% | 0.1734 | 94.84% |
| 6 | 0.0435 | 98.98% | 0.1822 | 94.84% |
| 7 | 0.0360 | 99.46% | 0.1758 | 94.70% |
| 8 | 0.0269 | 99.73% | 0.1649 | 95.38% |
| **9** | **0.0227** | **99.69%** | **0.1621** | **95.52%** |
| 10 | 0.0194 | 99.73% | 0.1742 | 95.24% |
| 11 | 0.0135 | 99.90% | 0.1675 | 95.11% |
| 12 | 0.0121 | 99.97% | 0.1684 | 95.11% |
| 13 | 0.0107 | 99.93% | 0.1734 | 95.24% |
| 14 | 0.0083 | 100.00% | 0.1686 | 95.38% |
| 15 | 0.0083 | 99.90% | 0.1690 | 95.24% |
| 16 | 0.0066 | 99.97% | 0.1749 | 94.57% |
| 17 | 0.0070 | 99.90% | 0.1742 | 94.97% |
| 18 | 0.0049 | 100.00% | 0.1804 | 95.11% |
| 19 | 0.0047 | 100.00% | 0.1773 | 94.84% |
| 20 | 0.0030 | 100.00% | 0.1751 | 94.84% |

### 7.2 最佳验证结果

| 指标 | 结果 |
|---|---:|
| 最佳 epoch | 9 |
| 最佳验证准确率 | 95.52% |
| 对应验证损失 | 0.1621 |

### 7.3 最终测试结果

恢复第 9 个 epoch 的最佳检查点后，测试结果为：

| 指标 | 结果 |
|---|---:|
| 测试集样本数 | 3669 |
| 正确分类数 | 3426 |
| 错误分类数 | 243 |
| 测试损失 | 0.2254 |
| **Top-1 测试准确率** | **93.38%** |

最佳检查点保存在：

```text
D:\ViT-Transfer-Learning\checkpoints\lora\best_model.pt
```

文件大小为 1,309,922 字节，约 1.25 MB。

## 8. 结果分析

### 8.1 收敛情况

模型在第一个 epoch 后即达到 91.85% 的验证准确率，说明 ImageNet 预训练表示已经能够较好地区分不同宠物品种。随着 LoRA 和分类头继续训练，验证准确率在第 4 个 epoch 达到 95.11%，并在第 9 个 epoch 达到最高值 95.52%。

### 8.2 过拟合现象

第 9 个 epoch 以后，训练准确率继续提高并最终达到 100%，训练损失持续下降至 0.0030；但验证准确率没有同步提高，而是在约 94.6%～95.4% 范围内波动，验证损失也不再持续下降。这表明模型在后半段出现轻微过拟合。

最佳检查点选择机制避免了直接使用最后一个 epoch 的参数。最终测试所使用的是第 9 个 epoch，而不是第 20 个 epoch。

### 8.3 参数效率

本实验只训练完整模型约 0.375% 的参数，任务检查点约 1.25 MB。与为每个任务保存一份完整 ViT-B/16 权重相比，LoRA 任务参数的存储成本显著降低。

需要注意的是，LoRA 主要减少可训练参数、梯度和优化器状态，并不会省去完整 backbone 的前向与反向计算。训练时仍需通过全部 12 个 Encoder Block 将梯度传播至各层 LoRA 分支。

### 8.4 当前结论边界

当前实验可以支持以下结论：

> 在 Oxford-IIIT Pet 37 类分类任务上，冻结 ViT-B/16 原始参数，仅训练所有 Encoder Block 中 Query/Value 的 rank-8 LoRA 和新分类头，可以获得 93.38% 的单次测试准确率。

当前实验不能支持以下结论：

- LoRA 一定优于只训练分类头；
- rank 8 是最优 rank；
- Query 和 Value 是最优注入位置；
- LoRA 一定优于全参数微调；
- 93.38% 是稳定的多次实验均值。

## 9. 局限性

1. **只有单次对比运行**：LoRA 与 Linear Probe 当前都只使用随机种子 42，没有报告均值和标准差。
2. **缺少 rank 消融**：尚未比较不同低秩维度。
3. **缺少注入位置消融**：尚未比较仅 Query、Query+Value、Query+Key+Value 或 Attention+MLP。
4. **缺少 OFT**：当前还不能形成完整的四方法比较。
5. **训练策略较基础**：未使用学习率调度、早停、label smoothing 等技术。
6. **缺少类别级分析**：尚未统计每类准确率、混淆矩阵和典型错误案例。
7. **任务检查点依赖基础权重**：1.25 MB 检查点只包含 LoRA 与分类头，恢复模型时仍需加载相同的 ViT-B/16 预训练权重。

## 10. 后续实验计划

建议按照以下顺序继续：

1. **OFT**：加入保持正交结构的参数高效微调方法。
2. **LoRA rank 消融**：比较 \(r\in\{2,4,8,16\}\)，保持其他条件不变。
3. **LoRA 注入位置消融**：比较 Query、Query+Value、Query+Key+Value。
4. **重复实验**：至少使用 3 个随机种子，报告测试准确率均值和标准差。
5. **类别级评估**：生成混淆矩阵、每类准确率和错误样本分析。

Linear Probe 与 Full Fine-tuning 已经完成。下一步应加入 OFT，形成四种迁移策略的统一比较；随后再进行 LoRA rank 和注入位置消融。

## 11. 项目代码结构与复现

| 文件 | 功能 |
|---|---|
| `scripts/download_pet_dataset.py` | 下载 Oxford-IIIT Pet 数据集 |
| `scripts/download_pretrained_model.py` | 下载 ViT-B/16 预训练权重 |
| `experiments/lora/model.py` | ViT、LoRA和分类头结构 |
| `vit_core/pretrained.py` | 统一映射并加载预训练权重 |
| `vit_core/pet_dataset.py` | 数据划分、预处理和 DataLoader |
| `vit_core/training_engine.py` | 单 epoch 与多 epoch 训练流程 |
| `vit_core/evaluation.py` | 验证、测试与任务检查点加载 |
| `experiments/lora/run_lora.py` | 完整实验入口 |
| `tests/test_lora.py` | 不启动完整训练的端到端冒烟测试 |

正式运行命令：

```powershell
cd D:\ViT-Transfer-Learning
E:\anaconda\python.exe -m experiments.lora.run_lora
```

恢复最佳模型并进行测试时，需要先构造并加载预训练模型，再加载任务检查点：

```python
from experiments.lora.model import ViTForPetClassification
from experiments.lora.run_lora import CHECKPOINT_PATH
from vit_core.checkpoint import load_trainable_checkpoint
from vit_core.pretrained import load_pretrained_vit

model = ViTForPetClassification()
load_pretrained_vit(model)
load_trainable_checkpoint(model, CHECKPOINT_PATH)
```

## 参考资料

1. Dosovitskiy, A. et al. [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR, 2021.
2. Hu, E. J. et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR, 2022.
3. Parkhi, O. M. et al. [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf). CVPR, 2012.
4. Google. [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224). Hugging Face model repository.
