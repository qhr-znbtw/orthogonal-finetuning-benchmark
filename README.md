# ViT-LoRA for Oxford-IIIT Pet Classification

一个从零实现核心网络结构的参数高效迁移学习项目：把 ImageNet 预训练的 ViT-B/16 迁移到 Oxford-IIIT Pet 37 类细粒度分类任务，并且仅训练 Query/Value LoRA 分支与新的分类头。

## 实验结果

| 指标 | 结果 |
| --- | ---: |
| 最佳验证准确率 | **95.52%** |
| 官方测试集准确率 | **93.38%** |
| 测试集正确数 | **3426 / 3669** |
| 总参数量 | 86,122,021 |
| 可训练参数量 | 323,365 |
| 可训练参数占比 | **0.375%** |

实验使用单张 NVIDIA GeForce RTX 4070 Ti SUPER，训练 20 个 epoch；最佳验证结果出现在第 9 个 epoch。完整训练过程与结论见 [research_report.md](research_report.md)。

## 方法

- 主干：`google/vit-base-patch16-224`
- 输入：`224 × 224` RGB 图像
- ViT：12 个 Transformer Encoder Block，隐藏维度 768，12 个注意力头
- LoRA：加入全部 Encoder Block 的 Query 与 Value 投影
- LoRA rank：8
- LoRA alpha：8
- 分类头：`LayerNorm + Linear(768, 37)`
- 优化范围：仅 LoRA 参数和分类头

本项目直接实现 ViT 嵌入层、Multi-Head Self-Attention、MLP、Encoder Block 和 LoRA 线性层，并通过参数名映射加载 Hugging Face 发布的预训练权重，而不是直接调用完整的现成分类模型。

## 项目结构

```text
.
├── dataset.py                    # 数据划分与 DataLoader
├── download_pet_dataset.py       # 下载 Oxford-IIIT Pet
├── download_pretrained_model.py  # 下载 ViT-B/16 预训练权重
├── load_pretrained_model.py      # 映射并加载预训练参数
├── main.py                       # 完整训练与测试入口
├── net.py                        # ViT-B/16 与 LoRA 实现
├── test.py                       # 验证/测试逻辑
├── train.py                      # 单轮及多轮训练逻辑
├── research_report.md            # 完整研究报告与逐轮结果
└── requirements.txt
```

数据集、预训练模型与训练 checkpoint 不提交到 Git 仓库，可通过脚本下载或在本地重新生成。

## 环境安装

建议使用 Python 3.10 或更高版本，并根据本机 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配的 PyTorch。其余依赖可执行：

```bash
pip install -r requirements.txt
```

## 复现实验

1. 下载 Oxford-IIIT Pet：

   ```bash
   python download_pet_dataset.py
   ```

2. 下载预训练 ViT-B/16：

   ```bash
   python download_pretrained_model.py
   ```

3. 开始训练并在最佳检查点上测试：

   ```bash
   python main.py
   ```

最佳模型默认保存至 `checkpoints/best_model.pt`。固定随机种子为 42，训练集与验证集从官方 `trainval` 集按类别进行 8:2 分层划分，官方测试集只用于最终评估。

## 训练配置

| 超参数 | 数值 |
| --- | ---: |
| Epoch | 20 |
| 训练 batch size | 32 |
| 验证/测试 batch size | 64 |
| LoRA 学习率 | `1e-4` |
| 分类头学习率 | `1e-3` |
| Weight decay | `0.01` |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |

## 当前结论与边界

当前结果表明，该实现能够以不到 0.4% 的可训练参数完成有效的细粒度宠物分类。当前仓库记录的是一次完整 LoRA 实验，尚未完成“冻结主干、只训练分类头”的 baseline、不同 rank 的消融实验和多随机种子重复实验，因此不能仅凭现有结果宣称 LoRA 优于其他迁移方法。

## 后续计划

- 增加 frozen-backbone + linear head baseline
- 比较 LoRA rank `r = 2, 4, 8, 16`
- 比较只适配 Query、只适配 Value 与同时适配 Query/Value
- 增加全参数微调结果
- 报告多随机种子的均值与标准差

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

