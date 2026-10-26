# ViT-B/16 BOFT 宠物品种分类实验报告

## 1. 实验目标

本实验使用 BOFT（Butterfly Orthogonal Fine-Tuning）将 ImageNet 预训练的
ViT-B/16 迁移到 Oxford-IIIT Pet 37 类分类任务。为与现有 rank-8 LoRA 实验保持
目标位置一致，BOFT 只注入全部 12 个 Transformer Encoder Block 的 Query 和
Value 投影；原始 ViT 参数保持冻结，新的 37 类分类头参与训练。

对于每个冻结权重

\[
W\in\mathbb R^{768\times768},
\]

BOFT 使用

\[
W'=WR_{\mathrm{BOFT}}^T,
\qquad
R_{\mathrm{BOFT}}=B_6B_5\cdots B_1.
\]

每个 butterfly factor 含 32 个独立的 \(24\times24\) 正交块。相邻 factor 使用
不同的固定置换，六层后的有效覆盖范围依次为

\[
24\rightarrow48\rightarrow96\rightarrow192\rightarrow384\rightarrow768.
\]

## 2. 参数化与正确性

每个正交块由反对称矩阵 \(Q\) 通过 Cayley 变换产生：

\[
R=(I-Q)(I+Q)^{-1}.
\]

实现只存储 \(Q\) 的严格上三角独立元素，而不保存冗余的完整矩阵 \(A\)。因此每个
\(24\times24\) 块包含

\[
\frac{24\times23}{2}=276
\]

个可训练自由度。单个 Query 或 Value 投影的 BOFT 参数量为

\[
6\times32\times276=52,992.
\]

12 层 Query 和 Value 共包含

\[
52,992\times2\times12=1,271,808
\]

个 BOFT 参数。分类头包含 28,453 个参数。

| 参数统计 | 数量 |
|---|---:|
| 完整模型参数 | 87,098,917 |
| BOFT 参数 | 1,271,808 |
| 分类头参数 | 28,453 |
| 总可训练参数 | 1,300,261 |
| 可训练参数比例 | 1.4929% |

自动测试验证了：

1. 每个 Cayley block 满足 \(R_i^TR_i\approx I\)；
2. 六个 butterfly factors 的稠密乘积满足 \(R^TR\approx I\)；
3. 稀疏快速前向与显式权重 \(W'=WR^T\) 的结果一致；
4. 全零初始化时 BOFT 网络与冻结基础线性层输出一致；
5. 198 个 ViT 预训练张量能够正确加载到 BOFT 包装后的网络。

## 3. 训练设置

| 配置项 | 取值 |
|---|---:|
| 预训练模型 | `google/vit-base-patch16-224` |
| 数据集 | Oxford-IIIT Pet |
| 随机种子 | 42 |
| Epoch | 20 |
| 训练 batch size | 32 |
| 验证/测试 batch size | 64 |
| BOFT block size | 24 |
| Butterfly factors | 6 |
| BOFT 学习率 | `1e-4` |
| 分类头学习率 | `1e-3` |
| Weight decay | `0.01` |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |
| 训练精度 | BF16 自动混合精度 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |

## 4. 完整训练记录

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.7469 | 82.81% | 0.2152 | 93.75% |
| 2 | 0.1137 | 97.04% | 0.1950 | 93.89% |
| 3 | 0.0577 | 98.30% | 0.1939 | 93.75% |
| 4 | 0.0270 | 99.59% | 0.1702 | 94.70% |
| 5 | 0.0158 | 99.73% | 0.1939 | 94.43% |
| 6 | 0.0078 | 99.97% | 0.1854 | 94.29% |
| 7 | 0.0062 | 99.97% | 0.1900 | 94.43% |
| 8 | 0.0067 | 99.90% | 0.1832 | 94.16% |
| 9 | 0.0117 | 99.83% | 0.1873 | 94.16% |
| 10 | 0.0034 | 99.97% | 0.1818 | 94.02% |
| 11 | 0.0020 | 100.00% | 0.1874 | 94.02% |
| 12 | 0.0017 | 100.00% | 0.1925 | 94.02% |
| 13 | 0.0017 | 100.00% | 0.1970 | 94.16% |
| 14 | 0.0015 | 100.00% | 0.1953 | 94.57% |
| 15 | 0.0014 | 99.97% | 0.2011 | 94.02% |
| 16 | 0.0029 | 99.93% | 0.2037 | 94.29% |
| **17** | **0.0023** | **99.97%** | **0.1958** | **94.84%** |
| 18 | 0.0156 | 99.59% | 0.2498 | 93.61% |
| 19 | 0.0041 | 99.93% | 0.2198 | 94.29% |
| 20 | 0.0010 | 100.00% | 0.2163 | 93.89% |

## 5. 最终结果与耗时

训练结束后恢复第 17 个 epoch 的最佳验证检查点，再对官方测试集进行评估。

| 指标 | 结果 |
|---|---:|
| 最佳 epoch | 17 |
| 最佳验证准确率 | 94.84% |
| 测试损失 | 0.2656 |
| 测试准确率 | **93.51%** |
| 正确分类 | 3431 / 3669 |
| 20-epoch 训练墙钟时间 | **414.02 秒（6 分 54 秒）** |
| 最终测试墙钟时间 | 15.09 秒 |
| 最佳任务检查点大小 | 5,210,082 字节（约 4.97 MiB） |

训练计时覆盖每个 epoch 的训练与验证，以及训练结束后的最佳检查点恢复；测试计时单独
记录。计时结果来自本次单张 RTX 4070 Ti SUPER 的实际运行，并非理论估算。

## 6. 与现有结果的关系

在当前单随机种子运行中：

- BOFT 测试准确率 93.51%，比 Linear Probe 高 0.90 个百分点；
- BOFT 比 rank-8 LoRA 高 0.13 个百分点，即多正确分类 5 张图片；
- BOFT 比 Full Fine-tuning 高 0.27 个百分点，即多正确分类 10 张图片；
- BOFT 的最佳验证准确率 94.84%，低于 LoRA 的 95.52%。

BOFT 使用 1,300,261 个可训练参数，约为当前 LoRA 323,365 个参数的 4.02 倍。因此，
本次结果不能说明 BOFT 在相同参数预算下优于 LoRA，只能说明该 \(b=24,m=6\) 的完整
butterfly 配置能够获得有竞争力的测试性能。后续若要进行严格方法比较，需要设计匹配
参数预算的 BOFT 配置，并使用多个随机种子重复实验。

## 7. 复现

```powershell
cd D:\ViT-Transfer-Learning
E:\anaconda\python.exe -m tests.test_boft
E:\anaconda\python.exe -m experiments.boft.run_boft
```

机器可读结果保存在 `results/boft.json`，最佳任务参数保存在
`checkpoints/boft/best_model.pt`。

## 参考资料

1. Liu, W. et al. [Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization](https://arxiv.org/abs/2311.06243). ICLR, 2024.
2. Qiu, Z. et al. [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280). NeurIPS, 2023.
3. Dosovitskiy, A. et al. [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929). ICLR, 2021.
