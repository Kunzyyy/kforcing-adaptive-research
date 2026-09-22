# 作者核对补充：历史完整检查点与 EMA

未发送；这份材料补充此前草稿，不覆盖历史记录。需要发送时填写姓名、单位与回复方式。

庄老师及 K-Forcing 团队您好：

我们对公开LM1B权重的基线复现继续做了版本排查，在HF历史revision `6b357e2893399a3c0219df6ef62af8fc6af747a4` 找到了包含训练配置和EMA的完整检查点。希望核对它们与论文表1的关系。

1. 当前精简K4检查点SHA256为 `3b013f0cf9a3acab4c20a1bb748beee6525432ef3f56a74df4957b898480a2c2`。其109个状态张量与历史普通state_dict的形状、大小和CRC32一致；历史EMA的108个参数不同。表1实际使用普通权重、EMA，还是其他训练版本？能否给出对应hash或完整配置？
2. 历史K4字段为global_step=150364、trainer.max_steps=500000、EMA更新次数150364，并带有恢复训练配置。历史AR的global_step=999412。我们知道恢复训练可能使计数不代表累计步数；请问这些发布文件对应哪个训练阶段、哪套报告结果？
3. 历史EMA列表没有参数名。我们按state_dict参数顺序去掉rotary buffer恢复，全部形状/步幅/类型与发布模型named_parameters顺序匹配。旧训练中EMA是否以此顺序登记参数？
4. 能否提供表1的固定前缀、真实Gen-PPL评分入口、生成精度/温度/频率惩罚设置？论文RMSNorm与公开F.layer_norm的对应关系也希望确认。

作为定向排查，在旧128个前缀、batch4、相同种子和相同评分下，AR普通/EMA Gen-PPL为121.64/115.18，K4为256.79/267.39。两组差异的配对描述区间均跨零。当前这套EMA恢复没有解释K4差距，不能归因于作者错误，也未证明论文复现成功。旧256条评分精确重现。

我们没有更改旧权重、自适应模型或先前结论。元数据读取、EMA提取与小规模诊断的源码、原始样本和独立统计核对均已保存。希望这些具体信息能降低核对成本。谢谢！

## 附件索引

- [诊断报告](历史检查点与EMA诊断报告.md)
- [当前/历史存储校验表](results/public-history/storage_comparison.json)
- [K4历史元数据](results/public-history/historical_pflm/metadata_summary.json)
- [AR历史元数据](results/public-history/historical_ar/metadata_summary.json)
- [128源句统计](results/historical-ema/analysis.json)
- [独立审计](results/historical-ema/integrity_audit.json)
- [旧固定输入工具说明](results/baseline-016/author_probe/README.md)：该工具只检查当前发布普通权重，不直接检验本轮EMA映射。
