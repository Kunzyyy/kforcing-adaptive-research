# K-Forcing Adaptive Decoding Research

[![verify-evidence](https://github.com/Kunzyyy/kforcing-adaptive-research/actions/workflows/verify-evidence.yml/badge.svg)](https://github.com/Kunzyyy/kforcing-adaptive-research/actions/workflows/verify-evidence.yml)

K-Forcing自适应解码的实验代码、复现记录与数值核查材料。

**研究状态：探索阶段。当前尚未建立可靠的自适应质量/速度优势，公开基线与论文仍有未解释差距，人工质量评分为0。** 仓库保留成功的实现核对与未通过的研究假设，不把诊断信号写成已验证的新算法。

## 研究问题

冻结公开K-Forcing模型，在生成过程中判断哪里值得重算、每次提交多少候选。最终需要在全新数据上确认决策收益，并在完整生成中计入全部开销，检验文本质量与耗时的取舍。

## 当前主要结论

| 实验 | 结论与证据 |
|---|---|
| 基线、在线与后续确认 | 原方案尚无可靠优势；见[研究交付摘要](outputs/kforcing-research-handoff/研究交付摘要.md) |
| 注意力特征 | 新增6个最后层特征未通过开发门槛；见[报告](outputs/kforcing-adaptive/注意力决策特征实验报告.md) |
| 局部机制与目标迁移 | 真实局部教师评分包含排序信息，但固定19维模型换标签后未通过门槛；见[报告](outputs/kforcing-adaptive/局部机制与目标迁移实验报告.md) |

最近的注意力与局部机制实验复用了已有训练/开发队列，不能视为独立确认。历史独立确认与近期开发诊断的范围在各报告中分别说明。

## 五分钟CPU数值复核

本地验证环境：Python 3.12.10、NumPy 2.5.3。以下入口无需模型权重或GPU，重算保存证据中的特征、拟合、预测与统计区间：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-evidence.txt
.venv\Scripts\python scripts/verify_evidence.py
```

Linux/macOS将上面Python可执行文件换为`.venv/bin/python`。`passed: true`表示数值核对通过，不代表算法假设通过。此脚本只读取保存材料，不修改实验记录。

同一入口在每次推送和PR时由GitHub Actions自动执行（[verify-evidence](.github/workflows/verify-evidence.yml)，Python 3.12与3.14各跑一次，仅用`requirements-evidence.txt`中固定的NumPy）。页首徽章为绿，只说明保存证据能在干净环境中按位重算出相同数值；它不表示自适应方案取得优势，也不构成独立复现。每次运行的完整JSON报告可在该次Actions运行的summary与artifact中查看。

CPU核查与从零重新跑GPU实验是两个范围。后者需要额外模型、数据、依赖和相应计算时间，见[复现与目录说明](docs/REPRODUCIBILITY.md)。

## 目录

```text
outputs/
  kforcing-adaptive/        # 研究代码、各阶段方案、报告、保存结果、上游代码
  kforcing-research-handoff/ # 较早四组主要证据的精简复核材料
scripts/verify_evidence.py  # 最新三项实验的统一CPU核查入口
docs/                      # 复现、上传及发布范围说明
PUBLICATION_MANIFEST.json  # 从原工作区逐字节复制的文件SHA256
```

保留原`outputs/kforcing-adaptive`路径，是因为一些运行器从该位置定位仓库根目录下的`work`。不要仅把里面的Python文件平铺到仓库根目录。

较早阶段的ZIP是本地历史快照，未重复放入Git；历史报告中的ZIP链接属于原工作区归档引用。[完整研究索引](outputs/kforcing-adaptive/README.md)保留原记录。较早精简交付包不包含最近两轮实验，阅读时以本首页链接的新报告为准。

## 来源

- 官方上游：[alibaba-damo-academy/K-Forcing](https://github.com/alibaba-damo-academy/K-Forcing)
- 本研究使用的上游commit：`706caa332a69d509b7fc2fa53ac7819f381a8c78`
- 权重来源：`zwave/K-Forcing`，固定revision：`16b984316bdd4585f17b11d0e61a58d236c2febc`
- 本地修改和研究过程：[provenance.json](outputs/kforcing-adaptive/provenance.json)
- 上游许可证：[Apache-2.0](outputs/kforcing-adaptive/upstream/LICENSE)，详见[来源说明](THIRD_PARTY_NOTICES.md)。本项目不是原作者官方仓库。

GitHub仓库：[Kunzyyy/kforcing-adaptive-research](https://github.com/Kunzyyy/kforcing-adaptive-research)。后续更新见[GitHub上传指南](docs/GITHUB_UPLOAD.md)。
