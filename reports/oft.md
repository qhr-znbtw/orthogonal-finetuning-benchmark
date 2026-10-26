# ViT-B/16 原始分块 OFT 宠物品种分类实验报告

## 1. 实验目标

本实验使用原始分块 OFT（Orthogonal Fine-Tuning）将 ImageNet 预训练的
ViT-B/16 迁移到 Oxford-IIIT Pet 37 类分类任务。为与 LoRA 和 BOFT 保持一致，
OFT 只作用于全部 12 个 Transformer Encoder Block 的 Query 和 Value 投影；
预训练参数保持冻结，新的 37 类分类头参与训练。

对于每个冻结权重 \(W\in\mathbb R^{768\times768}\)，OFT 学习一个分块对角
正交矩阵 \(R\)，并使用

\[
W'=WR^T.
\]

本实验令 block size 为 24，因此 \(R\) 含 32 个相互独立的
\(24\times24\) 正交块。它与 BOFT 的核心区别是：OFT 只有一个分块对角
factor，不使用后续置换去扩大维度之间的信息覆盖范围。

## 2. 参数化与正确性

每个正交块由反对称矩阵 \(Q\) 通过 Cayley 变换生成：

\[
R=(I-Q)(I+Q)^{-1}.
\]

实现只保存 \(Q\) 的严格上三角独立元素，因此每个 \(24\times24\) 块有

\[
\frac{24\times23}{2}=276
\]

个可训练自由度。单个 Query 或 Value 投影含

\[
32\times276=8,832
\]

个 OFT 参数，12 层 Query 和 Value 总计

\[
8,832\times2\times12=211,968.
\]

| 参数统计 | 数量 |
|---|---:|
| 完整模型参数 | 86,039,077 |
| OFT 参数 | 211,968 |
| 分类头参数 | 28,453 |
| 总可训练参数 | 240,421 |
| 可训练参数比例 | 0.2794% |

自动测试验证了：

1. 每个 Cayley block 满足 \(R_i^TR_i\approx I\)；
2. 完整 OFT 矩阵具有严格分块对角结构且保持正交；
3. 快速前向与显式权重 \(W'=WR^T\) 的输出一致；
4. 198 个 ViT 预训练张量能够正确加载；
5. 将 24 个 Q/V 层合并为一次批量 Cayley 求解后，所有 OFT 参数仍能获得梯度。

为避免当前 Windows、PyTorch 2.5.1 和 CUDA 12.4 环境中大量重复小型
`torch.linalg.solve` 调用引发的 CUDA 后端异常，实现会在一次 ViT 前向开始时，
把 24 个 OFT 层的 Cayley 问题合并成一次批量求解。该优化不改变公式、参数或输出，
只减少底层求解器的调用次数。

## 3. 训练设置

| 配置项 | 取值 |
|---|---:|
| 预训练模型 | `google/vit-base-patch16-224` |
| 数据集 | Oxford-IIIT Pet |
| 随机种子 | 42 |
| Epoch | 20 |
| 训练 batch size | 32 |
| 验证/测试 batch size | 64 |
| OFT block size | 24 |
| OFT factors | 1 |
| OFT 学习率 | `1e-4` |
| 分类头学习率 | `1e-3` |
| Weight decay | `0.01` |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |
| 训练精度 | BF16 自动混合精度 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |

## 4. 完整训练记录

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.8332 | 81.18% | 0.2502 | 93.75% |
| 2 | 0.1661 | 96.06% | 0.2033 | 93.89% |
| 3 | 0.1032 | 97.28% | 0.1906 | 94.57% |
| 4 | 0.0650 | 98.54% | 0.1675 | 94.84% |
| 5 | 0.0440 | 99.05% | 0.1717 | 94.70% |
| 6 | 0.0299 | 99.52% | 0.1699 | 94.57% |
| 7 | 0.0245 | 99.73% | 0.1724 | 94.70% |
| 8 | 0.0183 | 99.86% | 0.1615 | 94.70% |
| 9 | 0.0151 | 99.86% | 0.1583 | 95.24% |
| 10 | 0.0130 | 99.86% | 0.1617 | 94.43% |
| 11 | 0.0089 | 99.93% | 0.1655 | 94.29% |
| 12 | 0.0080 | 100.00% | 0.1668 | 94.70% |
| 13 | 0.0071 | 99.97% | 0.1664 | 95.11% |
| 14 | 0.0054 | 100.00% | 0.1649 | 94.97% |
| 15 | 0.0062 | 99.90% | 0.1661 | 94.57% |
| 16 | 0.0045 | 100.00% | 0.1629 | 94.84% |
| **17** | **0.0038** | **100.00%** | **0.1647** | **95.38%** |
| 18 | 0.0033 | 100.00% | 0.1654 | 95.11% |
| 19 | 0.0034 | 100.00% | 0.1695 | 94.97% |
| 20 | 0.0020 | 100.00% | 0.1675 | 95.24% |

## 5. 最终结果与耗时

训练结束后恢复第 17 个 epoch 的最佳验证检查点，再对官方测试集进行评估。

| 指标 | 结果 |
|---|---:|
| 最佳 epoch | 17 |
| 最佳验证准确率 | **95.38%** |
| 测试损失 | 0.2471 |
| 测试准确率 | **93.13%** |
| 正确分类 | 3417 / 3669 |
| 20-epoch 训练墙钟时间 | **217.27 秒（3 分 37 秒）** |
| 最终测试墙钟时间 | 12.97 秒 |
| 最佳任务检查点大小 | 970,722 字节（约 0.93 MiB） |

## 6. 与其他方法的关系

在当前单随机种子实验中：

- OFT 比 Linear Probe 高 0.52 个测试百分点，多正确分类 19 张图片；
- OFT 比 rank-8 LoRA 低 0.25 个百分点，少正确分类 9 张图片；
- OFT 比六因子 BOFT 低 0.38 个百分点，少正确分类 14 张图片；
- OFT 比 Full Fine-tuning 低 0.11 个百分点，少正确分类 4 张图片。

OFT 的 240,421 个可训练参数少于 LoRA 的 323,365 个，也仅为 BOFT 的
18.49%。相应地，OFT 的实测训练时间为 217.27 秒，约为 BOFT 的 52.48%。
本次结果表现出清晰的容量—效率权衡：一个局部分块正交 factor 已经显著优于只训练
分类头；六因子 BOFT 提供跨 block 的全维混合后取得更高测试准确率，但付出了约
5.41 倍的可训练参数。由于目前只有一个随机种子，这些差异仍不能视为统计显著结论。

## 7. 复现

```powershell
cd D:\ViT-Transfer-Learning
E:\anaconda\python.exe -m tests.test_oft
E:\anaconda\python.exe -m experiments.oft.run_oft
```

机器可读结果保存在 `results/oft.json`，最佳任务参数保存在
`checkpoints/oft/best_model.pt`。

## 参考资料

1. Qiu, Z. et al. [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280). NeurIPS, 2023.
2. Liu, W. et al. [Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization](https://arxiv.org/abs/2311.06243). ICLR, 2024.
3. Dosovitskiy, A. et al. [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929). ICLR, 2021.
