# K-Forcing 研究证据交付包

2026-09-21。先读 [研究交付摘要](研究交付摘要.md)，再查看 `research_claims.json` 中的逐项证据位置。

本包汇总四类已有证据：基线复现差距、早期在线联合门槛失败、后来的19维离线确认失败，以及额外计算量的事后诊断。没有新增的自适应质量/速度成果，人工质量仍未评估。

## 最短复核路径

需要Python和NumPy，不需要CUDA、GPU、模型权重、PyTorch、Transformers或原始LM1B数据下载。本地验证使用Python3.12.10、NumPy2.5.3；其他平台/版本尚未实测。

已装NumPy时，进入解压目录后运行：

```sh
python verify_research_evidence.py
```

若环境缺少NumPy，可在自行准备的虚拟环境安装记录版本：

```sh
python -m pip install -r requirements-evidence.txt
python verify_research_evidence.py
```

依赖安装可能联网；复核脚本本身不访问网络。也可只检查文件哈希与摘要中的证据引用：

```sh
python verify_research_evidence.py --hashes-only
```

成功时输出JSON，包含 `passed: true`，以及baseline、online、offline_confirmation、compute_budget四组结果。缺文件、哈希不符、数值/区间不符时非零退出。失败不能通过手工改写哈希清单来忽略，应先确认文件来源。

程序读取原证据，不覆盖它们；复用旧计算量核查程序时只在系统临时目录操作，完成后清理。当前目录不生成新的模型输出或“人工评分”。

## 复核范围

| 入口 | 实际重算 | 不包含 |
|---|---|---|
| baseline | 1,024前缀×2方法的记录分数聚合、Gen-PPL及源句区间 | 重新生成、重新运行GPT-2、验证论文原环境 |
| online | 校准概率选择、1,536条方法/种子输出的记录计数、4,608次计时聚合、两项校正区间与原门槛 | 在新机器重新测延迟，或将结果归于后来的19维模型 |
| offline_confirmation | 681状态的八次平均标签、冻结排序、三项校正区间与原门槛 | 重训练、改变阈值、增加测试样本 |
| compute_budget | 10,896条分支计数、成本分解、归一化及116个区间 | 精确FLOPs、实测同耗时比较或可部署校准 |

哈希只能检查文件一致性，不能独立证明记录真实性、冻结时间或新颖性。原始神经网络推理与评分的复现需要完整项目、固定权重及相应硬件。

## 文件安排

- `研究交付摘要.md`：主要结论、边界、对实验室的用途和未完成项。
- `research_claims.json`：每项结论对应的JSON文件、字段和值；不合并不同实验的样本量。
- `results/`：四项结论用到的既有原始记录、统计及协议证据。
- `verify_research_evidence.py`：统一CPU复核入口。
- `verify_compute_budget.py`、`audit_compute_budget.py`：原有成本核查代码，按原字节保留。
- `PAYLOAD_SHA256.json`、`SOURCE_ORIGINS.json`：包内哈希和原项目相对路径；不是数字签名。
- `sources.json`、`checkpoint_manifest.json`：上游来源及固定版本；包内没有权重。
- `review_status.json`：打包时的人工评价状态快照，不会自动更新。

**不要把此研究包给未来的盲评人选。** 原始记录含方法名称及生成文本，可能破坏盲法。人工评审应只使用另一个包 `kforcing-blind-review.zip`；本包不包含那个页面、题目或解盲key。

本包未对外发送。作者复现配置询问材料和完整研究代码另行保存，不要求作者先阅读所有历史阶段记录。
