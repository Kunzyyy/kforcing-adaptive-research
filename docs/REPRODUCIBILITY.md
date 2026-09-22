# 复现与目录说明

## 已验证的CPU入口

在仓库根目录运行`python scripts/verify_evidence.py`。它顺序执行：

1. `outputs/kforcing-adaptive/verify_attention_features.py`
2. `outputs/kforcing-adaptive/verify_local_mechanism.py`
3. `outputs/kforcing-adaptive/verify_local_target_transfer.py`

只依赖Python和NumPy，核对保存的注意力、局部教师logits、模型拟合、折外选择与统计区间。它们不重新加载神经网络，不产生新续写或新的独立确认结果。

较早四组主要证据另位于`outputs/kforcing-research-handoff`，其README提供对应复核入口。

## GPU实验

研究源码和原始协议位于`outputs/kforcing-adaptive`。完整运行曾使用Windows、RTX 5060 Laptop GPU、PyTorch 2.11.0+cu128、Transformers 4.55.0；各实验的环境锁及精度记录以对应results文件为准。

模型和原始数据放在仓库根目录的以下本地路径，均不提交Git：

- `work/checkpoints/`：K-Forcing PFLM及AR模型。
- `work/gpt2-large/`：GPT-2-Large评分器。
- `work/lm1b-data/`：原始数据。
- `work/pilot-data/`：早期数据及词表。
- `work/kforcing-env/`：原工作区的Python环境，不随仓库复制。

下载和数据准备代码保留在研究目录。权重版本与SHA256见`provenance.json`及各实验manifest。先按具体实验协议安装依赖、准备文件；不要仅凭CPU核查通过就认为GPU环境已经配置完成。

原运行器通常拒绝覆盖已存在的结果。重新执行完整实验应另设输出目录或使用独立工作副本，并遵循对应协议；不要删除证据后覆盖生成。此发布整理没有修改原实验源码和结果，也未声称所有历史脚本都已在新环境从零重跑。

## 保存范围

保留研究代码、计划、报告和实验记录。模型权重、环境、原始parquet、缓存及重复ZIP不在本仓库。人工盲评`private/key.json`不发布，待收集的个人评分也应留在本地。因此涉及该private文件的旧完整清单或盲评核查不能用此发布副本独立完成；最新三项CPU入口不依赖它。

科研目录中的历史哈希清单来自原工作区，保留其原文；它们不是本发布包的成员列表。本发布包的复制文件清单使用根目录`PUBLICATION_MANIFEST.json`。文件按原字节保留，`.gitattributes`关闭自动换行转换，避免不同系统checkout导致冻结哈希失配。

本仓库包含带方法标签的研究输出，不能作为盲评人的评审材料。评审时使用原工作区单独的评审包。
