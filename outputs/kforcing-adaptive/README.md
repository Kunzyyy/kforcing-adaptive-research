# K-Forcing 自适应步长：研究实验记录

## 研究问题

冻结官方 LM1B PFLM 权重，探索计算前选择窗口和计算当前候选后选择提交长度两种自适应方式。

当前实现只是探测性基线。它不代表新方法已成立、超越作者成果、保持教师分布，或达到发表标准。

## 最新状态：局部教师线索有信息，固定标签迁移未通过

先读[局部机制与目标迁移实验报告](局部机制与目标迁移实验报告.md)。在既有446状态、253源句上完成两项先后独立冻结的实验。机制诊断中，真实局部教师排序相对原19维的完整续写收益差为+0.05147，95%描述区间[0.01461, 0.08624]；它读取重算后的候选，不能直接部署。随后固定原19维ridge，仅把训练标签换成该局部收益，候选相对原模型为−0.05602，97.5%区间[−0.10395, −0.00940]，推进条件失败。892个局部分支评分、30次拟合、2,676个预测及15个区间的数值核查通过。停止这组标签迁移；有效决策方法仍未成立，没有新测试、在线计时或人工评分。当前清单为`local_mechanism_file_hashes.json`。

## 注意力决策特征回顾

先读[注意力决策特征实验报告](注意力决策特征实验报告.md)。在既有446状态、253源句的训练/开发数据上，固定增加6个最后层注意力特征；只运行一个25维候选，同折对照14/19维。相对19维的选择收益差为−0.00694，97.5%区间[−0.01907, 0.00332]，相对简单规则区间也跨零，预设推进条件未通过。45次拟合、4,014个折外预测、14个统计区间及注意力数值重建复核通过。按计划停止这组实现，不启动新测试或继续调参。原测试结果未用于本轮选择；这是多次复用开发集上的筛选，不是独立确认。核心算法优势仍未成立，人工评分仍为0。当前清单为`attention_features_file_hashes.json`。

## 精简研究交付包回顾

先读 [研究交付摘要](../kforcing-research-handoff/研究交付摘要.md)，精简包为 [kforcing-research-handoff.zip](../kforcing-research-handoff.zip)，约5.9 MB、39个文件。CPU复核只需Python和NumPy：从ZIP解压到独立目录后，四组主要结论、1362个冻结模型预测与成本诊断116个区间均复核通过；文件修改与错误摘要引用的拒绝测试也通过。包内按实验区分早期在线hidden82、后来的19维离线确认、基线及事后成本诊断，不将不同模型结果混用。这是已有证据的交付整理，没有新模型实验或成功算法声明。用户暂时没有评审人选，人工评分仍为0；作者核对材料与研究包均未发送。研究包含带方法标签的输出，不给未来盲评人选阅读。当前清单 `research_handoff_file_hashes.json`，验证记录见 `results/research-handoff/validation.json`。

## 额外计算量归一化诊断回顾

先读 [计算预算公平性诊断报告](计算预算公平性诊断报告.md)。沿用681状态、380源句的冻结排序，不重新训练或生成。同选340个状态时，19维比14维增加的调用/候选更多；按相同期望额外调用或候选归一化后，19维相对14维评分收益差为−0.0003 / −0.0119，两项95%描述区间均跨零。10,896条分支计数、116个区间与旧主要点估计单独复核一致。该事后诊断未测耗时、不能改写原确认失败；人工评分仍为0，原盲评包保持不变。当前清单 `compute_budget_file_hashes.json`。

## 文本质量盲评工具回顾：人工结果待收集

先读 [文本质量盲评准备报告](文本质量盲评准备报告.md)。从既有决策确认队列按预先保存的方案随机抽取64个不同源句，加入16个检查项，生成80题离线评审包。抽样未读取标签或预测；方法映射与评审材料分开。128段文本、抽样及顺序复核一致，14项统计测试和15项浏览器检查通过。当前人工评分为0，质量仍为未评估，没有新的自适应优势。下一步需至少两名独立评审返回完整JSON；此前不再重复同类基线排查。评审只使用外层 `kforcing-blind-review.zip`；完整研究归档含解盲key，不应提供给评审。当前清单 `blind_quality_file_hashes.json`。

## 历史配置与分词器兼容性核查回顾

先读 [历史配置与分词器核查报告](历史配置与分词器核查报告.md)。历史AR/PFLM与当前30522项词表相同；1024源句编码、2048条decode均一致。当前普通权重分别按历史/当前配置实例化后，200固定输入的logits、10批缓存以及64对完整生成全部一致。完整性与独立CPU构造检查通过；未重算质量或修改旧算法。当前环境中这些字段和分词方式未解释论文差距。已整理[最小核对材料](最小复现核对材料.md)及独立复现包，未发送。仍无自适应质量/速度新优势。当前清单 `config_tokenizer_file_hashes.json`。

## 历史检查点与EMA诊断回顾

先读 [历史检查点与EMA诊断报告](历史检查点与EMA诊断报告.md)。核对27个官方代码祖先提交、11个模型revision后，找回旧完整检查点的训练配置与EMA。当前精简参数与历史普通参数存储校验一致；按顺序恢复EMA后完成128旧源句定向对照。K4普通/EMA Gen-PPL为256.79/267.39，比值95%描述区间[0.896,1.197]，EMA没有改善本轮基线。512条评分与独立审计通过，旧256条评分精确复现。已增加训练版本/EMA/保存步数核对材料，未发送。论文差距仍未解决，自适应仍无可靠新优势。当前清单 `historical_ema_file_hashes.json`。

## 训练目标对照回顾

先读 [训练目标对照报告](训练目标对照报告.md)。只在旧train/dev的446状态、253源句上比较完整标签与等长标签，固定14维ridge、分折和预算。原评分精确复现；等长标签两半相关更高，但对原完整续写任务的选择收益相对原标签为−0.0100，95%描述区间[−0.0287, 0.0064]，未达到开发推进标准，因此没有启动新测试或换参数。30次拟合及主要统计独立复核通过。保留原训练目标；论文基线差距仍未解决。当前清单 `training_target_file_hashes.json`。

## 决策特征与全新确认回顾

先读 [决策特征与新数据确认报告](决策特征与新数据确认报告.md)。开发集仅增加生成位置和EOS信息，在四组固定ridge对照中得到19维候选；随后冻结两模型、抽取384个全新源句并在评分前锁定预测。新测试681状态、380源句，19维候选未通过对随机、原14维和启发式的三项校正确认标准。两份独立审计通过。仍无完整在线速度或独立人类质量结论，论文基线差距未解决；当前快照清单 `decision_features_file_hashes.json`。

## 评分目标回顾审计

先读 [评分目标审计报告](评分目标审计报告.md)。重评分8,810条唯一续写与旧总分完全一致；阶段15的381源句、682状态在相同评分长度下仍有+0.2546平均GPT-2 NLL收益，AR教师也同向。重算未使文本整体变短，平均增加1.040次前向调用。该回顾分析保留重算问题的研究价值，但现有预测器仍未通过原独立确认，论文基线差距未解决。独立统计复核通过；[作者核对草稿](给作者的复现核对草稿.md)已备好，尚未发送。最新快照清单为 `reward_audit_file_hashes.json`。

## 第十六阶段回顾

先读 [第十六阶段实验报告](第十六阶段实验报告.md)。独立权重数学前向核对400个候选全部一致；1024新源句的AR/K4 Gen-PPL为114.39/215.71，K4与论文127.6的差距仍未解释。旧512条评分精确重现，源句/统计/重放审计通过。发布代码LayerNorm与论文表4的RMSNorm描述待核对；已准备未发送的作者固定输入工具。当前清单 `stage16_file_hashes.json`。

## 第十五阶段回顾

先读 [第十五阶段实验报告](第十五阶段实验报告.md)。384个全新源句确认两个预先冻结的14维候选；评分前锁定预测，与随机和简单置信度规则进行四项校正比较。两个低维候选均未通过预设离线确认门槛。独立审计通过。GPT-2仍是训练代理，无在线提速或独立人类质量结论。当前清单 `stage15_file_hashes.json`。

## 第十四阶段回顾

先读 [第十四阶段实验报告](第十四阶段实验报告.md)。只用阶段13训练/开发数据的446状态、253源句，完成18配置的重复分组嵌套交叉验证、199次条件标签置换和学习曲线。弱正则82维模型显示过拟合迹象；14维置信度配合较强约束出现探索性排序信号。没有重新评价历史测试，候选仍需全新源句确认。独立直接拟合审计通过，当前清单 `stage14_file_hashes.json`。

## 第十三阶段回顾

先读 [第十三阶段实验报告](第十三阶段实验报告.md)。384 个新源句、5,336 对分支，比较相同训练状态和共同配置下的八次平均标签模型与八个单次标签模型。开发集选择 hidden82_alpha001，九个模型冻结后再生成测试。两个主要测试区间都跨零，平均标签未显示可靠排序优势，推进门槛未通过。独立审计通过，当前清单 `stage13_file_hashes.json`；下一步检查特征可预测性与泛化。

## 第十二阶段回顾

先读 [第十二阶段实验报告](第十二阶段实验报告.md)。96 个新源句、340 个有效固定状态，每状态八次未来续写，共 2,720 对分支。单次标签波动较大，两组四次平均收益仍有可重复相关（0.703）；八次平均可靠性估计 0.819，不是模型准确率。两个冻结预测器仍无可靠排序优势；本轮未训练、没有在线提升结论。独立噪声重建、标签和主要统计区间审计通过，当前清单 `stage12_file_hashes.json`。

## 第十一阶段回顾

先读 [第十一阶段实验报告](第十一阶段实验报告.md)。384 个新源句用于 train/dev/test，单次干预后比较完整续写 GPT-2 每序列 NLL。六候选按 dev 选定 confidence14_alpha010，参数冻结后再采集测试；预设推进门槛未通过。这是离线评分收益预测，GPT-2 已参与优化目标，不是新的在线速度或人类质量证明。独立审计通过，当前清单 `stage11_file_hashes.json`。

## 第十阶段回顾

先读 [第十阶段实验报告](第十阶段实验报告.md)。完成真正 EOS 停止的冻结预测器测试，32 个新校准源句、96 个新测试源句；1,536 条测试方法/种子输出、4,608 次计时。主随机概率仅按独立校准计算计数冻结，加入同输出重放估计决策成本。预设近似预算与性能联合门槛未通过。独立审计通过，当前清单为 `stage10_file_hashes.json`。

## 第九阶段回顾

先读 [第九阶段实验报告](第九阶段实验报告.md)。冻结内部表示预测器完成 64 个新源句、640 条完整生成、1,920 次计时和独立 GPT-2 评分。主比较要求评分与固定长度吞吐量同时改善，结果见报告；固定长度计时包括 EOS 后位置，相同完整序列预算不保证可见文本预算匹配。当前清单为 `stage9_file_hashes.json`，前八阶段归档保留。

## 第八阶段回顾

先读 [第八阶段实验报告](第八阶段实验报告.md)。冻结第七阶段预测器，在 384 个新源句、4,096 个窗口上进行短前缀重复与长上下文确认；两个经过比较校正的主要局部终点均通过。教师条件 NLL 旁证与同噪声匹配指标方向不完全一致，尚无外部文本质量或在线速度收益结论。模型未重新拟合，独立审计及 NLL 复核通过。当前文件清单为 `stage8_file_hashes.json`；前七阶段 zip 保留。

## 第七阶段回顾

先读 [第七阶段实验报告](第七阶段实验报告.md)。在 448 个新前缀、3,584 个窗口上完成小型收益预测器的训练/开发/锁定测试。内部表示预测器在独立测试上的教师耦合匹配指标有微弱排序信号，但开发门槛未通过，与单次随机对照的区间也跨零，按预定规则不进入在线解码。没有文本质量或速度提升声明。独立审计通过。第七阶段快照为外层 `kforcing-adaptive-stage7.zip`。

## 第六阶段回顾

先读 [第六阶段实验报告](第六阶段实验报告.md)。公开代码、权重版本仍与本地相同。完成 1,024 个教师耦合局部窗口：拆成 2+2 有少量匹配收益，但尾部置信度在同预算下仍未胜过随机选择。指标不是文本质量或分布距离。独立审计通过。第六阶段快照为外层 `kforcing-adaptive-stage6.zip`。

## 第五阶段回顾

先读 [第五阶段实验报告](第五阶段实验报告.md)。完成相同噪声下的 fp32/bf16/fp16 对照：256 条控制输出精确重现第四阶段，新增 512 条精度对照输出。三种评分方式共 3,072 条记录，独立审计通过。已测缓存与训练/推理窗口路径一致；精度切换和两项替代评分方式未解释 k=4 的质量复现差距，根因仍未定位。自适应方法尚无可靠优势。

[复现差异核对说明](复现差异说明_待作者核对.md)已整理，尚未对外发送。第五阶段快照为外层 `kforcing-adaptive-stage5.zip`。

## 第四阶段回顾

先读 [第四阶段实验报告](第四阶段实验报告.md)。完成 LM1B 官方采样路径的 1,792 条生成，以及真实参考与旧自适应输出的外部 GPT-2-Large 复评，共 2,240 条评分。18 组同噪声采样对照一致，独立数据审计通过。旧自适应规则仍未显示可靠优势；本机固定 k=4 的质量基线与论文存在待解释差距。batch=16 的约 3.50 倍固定位置吞吐量收益属于原作者固定步长方法。

第四阶段源码与结果快照保留在外层 `kforcing-adaptive-stage4.zip`。

## 第三阶段回顾

先读 [第三阶段实验报告](第三阶段实验报告.md)。当前候选块信号通过开发集筛选后，在新文章上未显示可靠泛化收益。完整实现了自适应提交，完成 124 对新局部分支、320 条独立生成和 960 次计时，计入特征、决策及丢弃候选的开销；仍未超过随机提交或固定 k=3。7 项新增测试和独立数据审计通过。

## 第二阶段回顾

先读 [第二阶段实验报告](第二阶段实验报告.md)。Windows 教师模型的进程退出异常已通过 eager 残差辅助函数规避；完整 200 条生成回归正常退出，token 序列和教师 NLL 与原始记录一致。新数据上完成 662 对局部分支与收益预测；上一轮统计量的预测器仍没有可靠胜过随机选择的证据。

第一阶段源码和结果快照保留在外层 `kforcing-adaptive-stage1.zip`，第一阶段报告中的“退出异常待解决”为当时状态。

`final_file_hashes.json`、`stage2_file_hashes.json`、`stage3_file_hashes.json` 是历史清单。`portable-helper-fix.patch` 基于第一阶段 zip 中的便携版本，可用于审查第二阶段执行路径改动。前三阶段 zip 保持不变。

## 第一阶段实验约定

- 固定 `k=2/3/4`、随机 `k=2/4`、自适应 `k=2/4` 共五个方法。
- 自适应只读取上一轮原始 logits 的平均 top1-top2 间隔；间隔小于阈值时选 2，否则选 4。首轮选 4。
- 阈值使用 WikiText-2 validation 上 fixed4 的间隔中位数。随机策略的短步比例使用 validation 上自适应运行结果。测试集不参与调参。
- 相同 prompt/seed 使用按 token 位置索引的随机噪声，避免不同循环次数造成随机数错位。
- 根据实际本轮窗口传入噪声，缓存只保存文本上下文；噪声提示不进入长期 KV。
- EOS 后停止并截断输出；不把填充计为生成 token。耗时包括 prefill、选择、缓存、EOS 检查和所有方法共有的诊断统计。模型加载、分词和离线教师评分不计入。
- 按随机顺序运行方法，降低固定执行次序引起的偏差。
- 教师 NLL、重复率、平均长度和 EOS 比例仅为诊断指标；NLL 更低不等于文本质量更好。尤其需警惕重复、模式坍缩和长度差异。
- WikiText-2 与 LM1B 训练领域不同；小样本结果仅适合决定下一步实验。正式结果需增加同分布 held-out 数据、独立质量评价、多种子、置信区间和相关工作比较。
- batch=1，禁止据此推断并发服务收益。

## 文件

- `第五阶段实验报告.md`、`STAGE5_PROTOCOL.md`、`复现差异说明_待作者核对.md`：固定噪声的精度/评分排查与剩余复现问题。
- `diagnose_precision.py`、`score_precision.py`、`export_precision_noise.py`、`audit_stage5.py`：精度对照、评分敏感性、噪声导出与独立审计。
- `第四阶段实验报告.md`、`STAGE4_PROTOCOL.md`：官方口径核查、LM1B 基线、独立质量评价与限制。
- `prepare_lm1b.py`、`download_gpt2_evaluator.py`：固定版本、哈希校验的数据与外部评分器下载。
- `audit_official.py`、`run_lm1b_baselines.py`、`run_recommended_penalties.py`：采样语义对照、官方固定步长、公开推荐惩罚补充。
- `external_gpt2_score.py`、`test_external_score.py`：GPT-2-Large 续写评分及掩码核查。
- `analyze_lm1b_audit.py`、`audit_stage4.py`：配对区间与标准库独立完整性审计。
- `第三阶段实验报告.md`、`STAGE3_PROTOCOL.md`：当前候选块实验、预定筛选门槛和结果。
- `current_features.py`、`fit_current.py`：当前块统计与只使用 train/dev 的控制器筛选。
- `prepare_fresh.py`、`evaluate_current_fresh.py`：新文章排除、局部分支与锁定规则评估。
- `commit_decoder.py`、`run_commit_benchmark.py`：实际计算候选后提交 2/4，包含完整在线成本和三次重复计时。
- `analyze_commit.py`、`audit_stage3.py`：文章聚类比较、数据隔离/预算/重复计时审计。
- `第一阶段实验报告.md`：第一阶段历史结果与局限。
- `第二阶段实验报告.md`、`STAGE2_PROTOCOL.md`：稳定性修复、新数据预测实验和冻结协议。
- `runtime_probe.py`：自然退出码的隔离探针，支持 baseline / nojit / portable 三组。
- `collect_benefit.py`：按文章划分数据，采集局部分支并离线评分。
- `analyze_benefit.py`：训练/开发集选择的岭回归、独立测试与文章聚类区间。
- `audit_stage2.py`：数据隔离、锁定模型预测与 200 条回归的独立检查。
- `core.py`：严格加载权重、解码和离线指标。
- `test_correctness.py`：独立旋转数学参考、因果前缀一致性、变步长缓存一致性、EOS 测试。
- `prepare_data.py`：公开 WikiText-2 验证/测试 split 与 BERT 词表下载。
- `run_pilot.py`：校准、独立测试和逐样本 JSONL。
- `analyze.py`：按 prompt 聚类的配对 bootstrap、汇总及样本查看。
- `diagnose_signal.py`：仅在验证集比较同一上下文的 4 与 2+2，默认 fp32 分支。
- `audit_numerics.py`：复查首轮诊断中发现的 bf16 窗口数值差异。
- `validate_results.py`：不用 GPU，独立校验记录完整性和汇总计算。
- `upstream/`：原作者代码的必要子集及 Apache-2.0 许可。便携适配见 `provenance.json`。
- `results/`：运行记录和结果。

## 环境与运行

本机从工作区根目录使用 `work/kforcing-env/Scripts/python.exe`。模型位于 `work/checkpoints`，数据位于 `work/pilot-data`。不修改用户系统 Python。

独立重建环境（PowerShell）：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe download_checkpoints.py --dest ../../work/checkpoints
.venv/Scripts/python.exe prepare_data.py --dest ../../work/pilot-data
.venv/Scripts/python.exe -m pytest -q test_correctness.py
.venv/Scripts/python.exe run_pilot.py --checkpoints ../../work/checkpoints --data ../../work/pilot-data --out results/pilot-new
.venv/Scripts/python.exe analyze.py results/pilot-new
.venv/Scripts/python.exe validate_results.py results/pilot-new
.venv/Scripts/python.exe diagnose_signal.py --checkpoints ../../work/checkpoints --pilot results/pilot-new
```

检查点从 https://huggingface.co/zwave/K-Forcing 下载；固定 revision 和 SHA-256 见 `checkpoint_manifest.json`。使用 `download_checkpoints.py --dest ...` 下载并校验。不使用 `weights_only=False`。

本地便携实现使用 PyTorch SDPA 和经数学测试的纯 PyTorch RoPE，不能与论文 Linux/H100/FlashAttention 的耗时直接比较。RoPE 适配不意味着已在 FlashAttention 本体上做逐位一致性验证。

全部已安装版本见 `results/environment-lock.txt`。Windows 默认 `KFORCING_HELPERS=eager`，保留 `KFORCING_HELPERS=script` 以受控复现旧执行路径；不更改系统环境变量。相应算术保持一致，测试不能证明所有环境都不存在原生运行库问题。

第二阶段复现（项目目录内，使用新的结果目录）：

```powershell
.venv/Scripts/python.exe runtime_probe.py --checkpoints ../../work/checkpoints --out results/runtime-new --suite portable --iterations 24
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider test_correctness.py test_benefit_analysis.py
.venv/Scripts/python.exe collect_benefit.py collect --checkpoints ../../work/checkpoints --data ../../work/pilot-data --out results/benefit-new
.venv/Scripts/python.exe collect_benefit.py score --checkpoints ../../work/checkpoints --out results/benefit-new
.venv/Scripts/python.exe analyze_benefit.py results/benefit-new
```

`analyze_benefit.py` 不覆盖已有锁定模型。重复同一数据的运行只能用于复现检查，不能当成新的独立证据。

第三阶段复现（项目目录内，新结果目录；使用包内第二阶段固定的 train/dev 记录）：

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider test_current.py test_commit.py
.venv/Scripts/python.exe fit_current.py collect --checkpoints ../../work/checkpoints --out results/current-new
.venv/Scripts/python.exe fit_current.py screen --out results/current-new
```

仅在 `screening.json` 中 `passed_gate=true` 时继续：

```powershell
.venv/Scripts/python.exe prepare_fresh.py results/current-new ../../work/pilot-data
.venv/Scripts/python.exe evaluate_current_fresh.py collect --checkpoints ../../work/checkpoints --root results/current-new
.venv/Scripts/python.exe evaluate_current_fresh.py score --checkpoints ../../work/checkpoints --root results/current-new
.venv/Scripts/python.exe evaluate_current_fresh.py analyze --root results/current-new
.venv/Scripts/python.exe run_commit_benchmark.py calibrate --checkpoints ../../work/checkpoints --root results/current-new
.venv/Scripts/python.exe run_commit_benchmark.py run --checkpoints ../../work/checkpoints --data ../../work/pilot-data --root results/current-new
.venv/Scripts/python.exe analyze_commit.py results/current-new
.venv/Scripts/python.exe audit_stage3.py results/current-new
```

上述命令重现同一批已评估文章，不能用于声称新的泛化证据。`online_samples.jsonl` 有 320 条独立输出；`online_timings.jsonl` 的 960 次运行包含三次重复，分析已经合并这些重复。

## 已更正的初始判断

原作者采样函数读取 `self.max_k`，但官方 CLI 加载模型时设置 `max_k=args.K`。因此，常规固定 k CLI 没有由这个细节导致的普遍多算问题。我们只为同一模型动态改 k 明确区分计算窗口和输出数量；不能把这算作发现作者基线错误。

## 文献边界

- K-Forcing: https://arxiv.org/abs/2606.10820
- AdaEDL: https://arxiv.org/abs/2410.18351
- SVIP: https://arxiv.org/abs/2411.18462
- Adaptive Block Diffusion: https://arxiv.org/abs/2606.29275

后面三项提示“按置信度调整生成长度”已有相邻研究。投机解码中教师验收的保证不能直接搬到没有验收环节的 PFLM。当前仅作相关工作线索，尚未完成全面新颖性审查。


## 第四阶段复现

以下命令在工作区根目录执行，使用已经准备好的隔离环境与官方检查点；下载需要网络。新结果目录必须为空，脚本会拒绝覆盖已生成记录。完整复评还读取 `results/current-003` 的历史输出（交付包已包含）。

```powershell
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/prepare_lm1b.py --dest work/lm1b-data --vocab work/pilot-data/vocab.txt --out outputs/kforcing-adaptive/results/lm1b-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/download_gpt2_evaluator.py --dest work/gpt2-large
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_official.py --out outputs/kforcing-adaptive/results/lm1b-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_lm1b_baselines.py --out outputs/kforcing-adaptive/results/lm1b-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_recommended_penalties.py --out outputs/kforcing-adaptive/results/lm1b-new
& work/kforcing-env/Scripts/python.exe -m pytest -q -p no:cacheprovider outputs/kforcing-adaptive/test_external_score.py
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/external_gpt2_score.py --root outputs/kforcing-adaptive/results/lm1b-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/analyze_lm1b_audit.py outputs/kforcing-adaptive/results/lm1b-new
Copy-Item -LiteralPath work/gpt2-large/manifest.json -Destination outputs/kforcing-adaptive/results/lm1b-new/gpt2_model_manifest.json
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage4.py outputs/kforcing-adaptive/results/lm1b-new
```

已有交付结果可直接运行最后一条审计，将 `lm1b-new` 改为 `lm1b-004`；该审计仅需 Python 标准库，不需要 GPU 或下载权重。GPT-2-Large 权重约 3.25 GB，不包含在 zip 中。


## 第五阶段复现

在工作区根目录运行，沿用已核验的本地权重。默认读取第四阶段 `results/lm1b-004`；无需重新下载。使用新的结果目录，脚本拒绝覆盖已有输出。

```powershell
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/diagnose_precision.py --out outputs/kforcing-adaptive/results/precision-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/score_precision.py --root outputs/kforcing-adaptive/results/precision-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/export_precision_noise.py outputs/kforcing-adaptive/results/precision-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage5.py outputs/kforcing-adaptive/results/precision-new
```

已有结果审计仅需标准库 Python：运行 `audit_stage5.py`，将目录改为 `outputs/kforcing-adaptive/results/precision-005`。导出的 `replay_noise.json` 保留原始 float32 数值；异环境逐步对照时可直接使用，不必依赖随机数生成器版本一致。新的生成脚本默认重新生成噪声，并以 fp32 逐 token 重现旧输出作为门槛。


## 第六阶段复现

在工作区根目录运行，使用新的输出目录。

```powershell
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/coupled_teacher_audit.py --out outputs/kforcing-adaptive/results/coupled-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/analyze_coupled_teacher.py outputs/kforcing-adaptive/results/coupled-new
```

现有记录可用标准库 Python 运行 `audit_stage6.py outputs/kforcing-adaptive/results/provenance-006`；该审计另核对交付包中保存的公开 API 快照。


## 第七阶段复现

在工作区根目录顺序运行，使用新的输出目录。需要现有本地检查点、LM1B parquet 和 BERT 词表；基础模型不训练。测试标签只能在模型锁定后采集。

```powershell
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/learn_local_gain.py prepare --out outputs/kforcing-adaptive/results/learned-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/learn_local_gain.py collect --split train --out outputs/kforcing-adaptive/results/learned-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/learn_local_gain.py collect --split dev --out outputs/kforcing-adaptive/results/learned-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/learn_local_gain.py fit --out outputs/kforcing-adaptive/results/learned-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/learn_local_gain.py collect --split test --out outputs/kforcing-adaptive/results/learned-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/learn_local_gain.py evaluate --out outputs/kforcing-adaptive/results/learned-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage7.py outputs/kforcing-adaptive/results/learned-new
```

已有结果可运行最后一条，将 `learned-new` 改成 `learned-007`；审计仅需 Python + NumPy，不需要 GPU。完整特征、分支、标签、预测和 64 维投影均在结果目录内。不要依据已看过的 test 重新选模型。


## 第八阶段复现

在工作区根目录顺序执行；使用新的结果目录。第七阶段模型、投影和特征代码保持不变，不调用任何拟合程序。

```powershell
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/confirm_local_gain.py prepare --out outputs/kforcing-adaptive/results/confirm-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/confirm_local_gain.py collect --out outputs/kforcing-adaptive/results/confirm-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/confirm_local_gain.py analyze --out outputs/kforcing-adaptive/results/confirm-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/verify_confirmation_nll.py outputs/kforcing-adaptive/results/confirm-new
& work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage8.py outputs/kforcing-adaptive/results/confirm-new
```

已有结果可运行最后一条，将目录改为 `confirm-008`；该审计仅需 Python + NumPy，不使用 GPU。GPU 的 NLL 复核另列为倒数第二条，不会覆盖已有核查记录。`local_confirmation_passed` 只指本轮两个局部匹配终点，不能解释成完整生成质量/速度通过。


## 第九阶段复现

需已有、经校验的检查点、LM1B parquet、词表及 GPT-2-Large；保持仓库根目录为工作目录。已有结果路径禁止覆盖，独立重跑请另传新目录。

```powershell
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_frozen_online.py prepare
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_frozen_online.py preflight
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_frozen_online.py run
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage9.py outputs/kforcing-adaptive/results/online-009 --generation-only
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_frozen_online.py score
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/analyze_frozen_online.py outputs/kforcing-adaptive/results/online-009
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/diagnose_online_budget.py outputs/kforcing-adaptive/results/online-009
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage9.py outputs/kforcing-adaptive/results/online-009
```


## 第十阶段复现

与之前阶段使用同一已校验的本地权重、词表和数据；原始结果禁止覆盖，重跑请另设 `--out` 目录。

```powershell
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_early_stop.py prepare
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_early_stop.py preflight
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_early_stop.py calibrate
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/run_early_stop.py run
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage10.py outputs/kforcing-adaptive/results/early-010 --generation-only
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/score_early_stop.py
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/analyze_early_stop.py outputs/kforcing-adaptive/results/early-010
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage10.py outputs/kforcing-adaptive/results/early-010
```


## 第十一阶段复现

保持仓库根目录为工作目录、已校验的基础权重与数据可用。结果文件禁止覆盖，新重跑请用新的 `--out` 路径。严格保持以下先后次序：

```powershell
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/rollout_quality.py prepare
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/rollout_quality.py preflight
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/rollout_quality.py collect --split train
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/score_rollout_quality.py --split train
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/rollout_quality.py collect --split dev
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/score_rollout_quality.py --split dev
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/fit_rollout_quality.py fit outputs/kforcing-adaptive/results/quality-011
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/rollout_quality.py collect --split test
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/score_rollout_quality.py --split test
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/fit_rollout_quality.py evaluate outputs/kforcing-adaptive/results/quality-011
work/kforcing-env/Scripts/python.exe outputs/kforcing-adaptive/audit_stage11.py outputs/kforcing-adaptive/results/quality-011
```
