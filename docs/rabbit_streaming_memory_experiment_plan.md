# 兔子行为长时上下文实验方案

## 目标与判断

目标是判断两件事：

1. 对已有 RepViT+GRU checkpoint，跨帧隐藏状态、状态重置和 FSM 各自带来什么效果。
2. 用连续视频训练 GRU，是否比现有独立短窗训练更能处理“进食后抬头咀嚼”和“跑动短暂停顿”。

训练与线上推理是联合实验变量，不预设必须复刻当前线上实现。FSM 作为单独的解码/状态机变量评估，不用它掩盖模型原始输出。

## 论文与官方实现参考

本实验优先复用论文中的训练/评估思想，不把人体动作数据集上的完整模型当成可直接部署到兔子模型的成品。

| 优先级 | 论文与实现 | 可借鉴内容 | 本项目的使用边界 |
|---|---|---|---|
| 1 | [MiniROAD, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/An_MiniROAD_Minimal_RNN_Framework_for_Online_Action_Detection_ICCV_2023_paper.html)；[作者实现](https://github.com/jbistanbul/MiniROAD) | 对同一流式 RNN 序列，比较不同时间步 loss weighting。论文的 step-function 方案只对序列末端 loss 计权，表现优于 uniform/linear/exponential/learnable；它针对训练/推理不一致。 | 作为本轮训练对照核心：在完全相同的流式输入与模型下比较逐步 uniform loss 与仅末步 loss。只移植/复写 criterion 思路到隔离实验目录；官方仓库的数据管线基于预提取 RGB/flow 特征和逐帧标签，不能直接替换本项目的 RepViT 图像 backbone 或目录数据集。保留论文/实现引用及许可证记录。 |
| 2 | [OnlineTAS, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/6c6c5fccf3c8661fcae219be7ca226f7-Abstract-Conference.html)；[作者仓库](https://github.com/QingZhong1996/OnlineTAS) | 在线分割的 adaptive memory、memory-enhanced features 和抑制 over-segmentation 的后处理，尤其适合作为跑—停—跑的分析参照。 | 作者仓库目前只有 README，标注实现 “Coming soon”，没有可复用的模型代码。因此本轮只记录机制并用现有 GRU/FSM 做可控消融，不以复现 OnlineTAS 为前置条件。 |
| 3 | [LSTR, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/08b255a5d42b89b0585260b6f2360bdd-Abstract.html)；[Amazon Science 官方实现](https://github.com/amazon-science/long-short-term-transformer) | 长期粗粒度历史 + 短期细节；代码提供 batch 与逐视频 stream 两种在线评估模式，可参考公平回放方式。 | 其代码假设预提取的高维 RGB/flow 特征、THUMOS/TVSeries 等数据和较重的 Transformer 训练管线。本轮不移植完整模型；若现有 GRU 经流式训练仍无法利用更长历史，再单独立项比较压缩历史特征头。 |
| 4 | [TeSTra, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1055_ECCV_2022_paper.php)；[作者实现](https://github.com/zhaoyue-zephyrus/TeSTra) | 用 box/Laplace temporal smoothing kernel 改造 streaming attention，使历史注意力可增量更新；它是在 LSTR 类 Transformer 上优化长记忆计算的路线。 | 代码依赖预提取特征和 LSTR 数据/模型组织，适合作为后续架构参考，不是现有 GRU 的轻量补丁；本轮只对比现有 GRU 与解码器，不引入它的依赖。 |

论文原实现中的 FPS/显存数字只描述各自论文硬件、数据和 backbone，不作为本项目的本地性能预期。

已核对的实现事实：当前训练的 `forward_seq()` 对一个 16 帧 clip 运行 GRU，只监督窗口末端；dashboard 的本地推理则逐帧更新 GRU hidden state，并按轨迹保存，目标超过 2 秒未更新会重置状态，另有约 5 秒候选状态 FSM。训练和线上因此不是同一运行方式。[训练模型](../models/timm_gru_student.py) [蒸馏循环](../training/distillation.py) [线上融合](../../handover/ppc_dashboard/backend/ml/decision/fusion_agent.py)

## 实验材料与冻结条件

### Baseline checkpoint

| 名称 | 权重 | 已有离线结果（30 条标注视频，0.5 秒采样） |
|---|---|---|
| B-Oct20 | `outputs/rabbit/2026-10-20/repvit_best_distilled_2026-10-20_ema.pth` | time-IoU macro 0.5558；event F1@tIoU0.5 0.4726；transition F1 0.0729 |
| B-Jun24 | `outputs/rabbit/2026-06-24/0624_repvit_best_distilled_ema.pth` | time-IoU macro 0.4242；event F1@tIoU0.5 0.3603；transition F1 0.0286 |

两者都是直接保存的 state dict，分类头均为 4 类，但文件里没有类别名或类别顺序。开始实验前必须从对应训练配置、原始预测 trace 或训练日志确认 `resting/active/eating/drinking` 的权重索引映射；无法确认时不得继续打分，也不得静默按当前配置猜测。

现有离线汇总来自训练仓库的独立短窗学生推理，不等于 dashboard 的在线逐帧 GRU+FSM。实验必须在新统一评估入口中重跑两个 checkpoint；上表只作历史参考。YOLO 权重、裁剪扩展比例、帧采样时间戳、图像预处理和类别映射全程固定。

### 数据集

- 训练/调参数据：`dataset/rabbit/2026-10-22/train/<class>` 与 `val/<class>`。
- 当前按扩展名扫描有 1,047 条 train、264 条 val 视频；隔离审计实际可解码 1,040 条 train、264 条 val。另发现 7 条 train MP4 没有可读视频流，必须修复源文件或从实验 manifest 明确排除并记录后才能训练，不能静默跳过。
- 使用前生成视频 manifest，至少包含视频路径、文件哈希、类别、原始录制/场次 ID、帧率、时长、split。按原始录制分组检查 train/val 泄漏；同一原视频的派生片段不得跨 split。
- 验证每个训练视频是否确实可以按其目录类别给整段逐时刻赋标签。若视频中存在标签定义不同的阶段，先修正标签/切段；不得因为文件夹标签而把不符合行为定义的帧强行赋成同一类。
- 现有 30 条兔子标注评测视频用于冻结测试。新 FSM 阈值和训练超参数只在 `val` 与单独标注的 challenge 集上确定；最终测试集只在方案选定后评估一次。该测试集已有历史结果，因此需注明不是此前完全未见的盲测集，不得据它反复挑参。
- challenge 集至少覆盖：低头咬草后抬头咀嚼、active 中短暂停顿后继续、相似静止/休息。由人工标注行为状态区间和关键事件边界，保留原始视频，不把它拆成独立 3 秒分类样本。

## 隔离实验目录与代码边界

所有本次实验新增或改写的 Python 脚本、配置、测试和实验专用推理/训练模块一律放在训练仓库的 `experiments/rabbit_streaming_memory/` 下。禁止修改训练仓库现有源码、配置、`.bat` 入口或 dashboard 业务代码；已有工作区改动保持原样。

建议目录：

```text
experiments/rabbit_streaming_memory/
  README.md                 # 单一实验入口、命令、依赖、结果位置
  configs/                  # baseline/stream 训练和回放配置
  src/                      # 训练、连续序列数据、回放与 FSM adapter
  tests/                    # 旧权重兼容、状态隔离、时间戳/损失测试
  references/               # 论文链接、来源仓库 commit、license、源文件 SHA256
  manifests/                 # 仅清单/标注元数据，不复制原视频
  outputs/                  # 本地结果；加入该子目录自己的 .gitignore
```

复用现有模型、指标或 dashboard 逻辑时，优先从实验包只读导入；若需要修改接口，则将最小必要源文件复制到 `src/` 后只改副本。每个副本在 `references/source_manifest.json` 记录原仓库路径、源文件 SHA-256、Git HEAD、工作区是否有未提交改动和复制日期。MiniROAD 只复写末步 loss weighting 所需的最小逻辑；不克隆整套数据/训练框架。若复制第三方代码，同时保留原许可证和版权声明。

README 提供从项目根目录执行的独立命令；实验脚本不得要求把实验模块导入生产包，也不得自动写入 `models/`、`training/`、`utils/`、`backend/` 或现有 checkpoint 目录。输出 checkpoint、日志、manifest 和评估结果均写在 `experiments/rabbit_streaming_memory/outputs/`。

### 实验包内的实现

1. 在 `src/data/` 新增连续视频数据接口，按时间戳顺序读取帧或 4 Hz 采样帧，返回逐时刻标签和视频边界；读取错误必须指出视频与帧号并失败，不返回全零伪样本。原有 `OnlineVideoDataset` 不动。
2. 将 `models/timm_gru_student.py` 复制到 `src/model/` 后，在副本抽出逐帧特征接口并提供 stream forward；保留模块/参数名以严格加载两个旧 checkpoint。测试整段 `forward_seq` 与逐步 stream logits 在 eval 模式数值一致。
3. 在 `src/train/` 实现新训练器：视频边界清空 hidden，视频内顺序递推；每块 12 秒监督长度（4 Hz 时 48 步），前置 5 秒 burn-in（20 步）不计损失；块间保留 hidden 并 detach。实现两种对照 loss：uniform 对所有监督步求平均；MiniROAD step-function 只对块末预测计分类损失。保持同一初始化、数据顺序、增强、优化预算和 teacher/KD 设置。
4. Streaming 训练禁用时间反转，只保留空间/颜色增强。checkpoint sidecar 记录类别顺序、采样率、crop/normalize、训练模式、loss weighting、源 checkpoint 哈希、随机种子和数据 manifest 哈希。
5. 在 `src/eval/` 新建统一回放器，显式支持 `window`、`stream` 和 FSM 模式；记录原始 logits/probabilities、raw argmax、最终解码、状态重置原因和分项耗时。指标可只读调用现有 `utils/metrics.py`；若接口不兼容则复制到 `src/` 后只修改副本。
6. 为复现 dashboard 行为，允许把 `offline_infer_rabbit.py` 及最小必要的识别器/FusionAgent 文件复制到 `src/dashboard_snapshot/` 并在副本中加入 checkpoint、FSM 和 reset 参数；生产仓库一律只读。保留按 camera/track 隔离 hidden 的行为，并记录轨迹创建、丢失、重连、重置时间戳。推理池不是本轮必要条件。

## 实验矩阵

所有回放使用同一视频、同一检测结果/裁剪、同一采样时间戳。每个 checkpoint 和训练候选分开跑以下解码器，避免把训练提升与 FSM 效果混在一起：

| 编号 | 模型/训练 | 推理方式 | 解码 |
|---|---|---|---|
| I0 | B-Oct20、B-Jun24 | 独立 16 帧因果滑窗，每 0.5 秒一次 | raw argmax |
| I1 | B-Oct20、B-Jun24 | 单帧 GRU hidden 连续递推 | raw argmax，FSM 关闭 |
| I2 | B-Oct20、B-Jun24 | 与 I1 完全相同 | 当前生产 FSM 和 2 秒 hidden-reset 规则 |
| I3 | B-Oct20、B-Jun24 | 与 I1 完全相同 | FSM 关闭，仅改变 hidden-reset 间隔；至少比较当前 2 秒与不因短暂停顿重置（只在轨迹结束/视频结束重置） |
| T0 | 从 B-Oct20 初始化，独立 16 帧短窗训练、只监督窗口末端 | `window` 和 `stream` 都评估 | raw argmax、固定 FSM |
| T1 | 同一初始化/数据/优化预算，连续流式训练、监督步 uniform loss | `stream` | raw argmax、同一固定 FSM |
| T2 | 同一初始化/数据/优化预算，连续流式训练、MiniROAD step-function 末步 loss | `stream` | raw argmax、同一固定 FSM |

T0/T1/T2 至少 3 个固定随机种子；逐种子成对比较。若算力预算不足，先完成 I0–I3 和一组 T0/T1/T2 验证数据管线，不能把单 seed 结果写成稳定结论。训练 epoch、学习率、batch/有效 batch、增强、teacher/KD、采样、早停与 checkpoint 选择规则必须相同。T1 与 T2 唯一差异是时间步 loss 权重；T0 与流式组用于比较训练序列和隐状态处理方式。

## 指标与判定

主指标沿用 `utils/metrics.py`：time-IoU macro、event F1@tIoU 0.3/0.5/0.7、transition F1、segment edit score、oscillation rate、短时错误状态持续时间 P90、boundary offset、late-commit latency P90、per-class event recall。另报告 eating→抬头咀嚼、active→短暂停顿两个 challenge 子集的错误切换次数/小时和恢复正确标签延迟。

性能基准全部在本地环境以离线视频回放完成，不依赖摄像头。记录 OS、CPU/GPU 型号、RAM/显存、驱动、Python/PyTorch/CUDA、推理后端、输入尺寸、batch=1、采样率和 detector 配置。每个候选先 warm-up 至少 100 次，再对同一组视频重复 5 轮；GPU 计时前后同步。分别报告纯模型和包含 decode/detector/crop 的端到端 p50/p95、持续处理帧率、CPU/GPU 占用及内存；耗时分项记录，不以零值代替。

2026-10-23 的首轮模型-only 合成输入基准：本机 Windows 11、RTX 4090、PyTorch 2.7.0+cu118、batch=1、224x224、100 次 warm-up、每模型 1,000 次计时。Oct20 单帧 stream p50/p95 为 7.20/11.01 ms，16 帧 `forward_seq` 为 7.07/10.80 ms；Jun24 分别为 6.99/10.60 ms 和 7.13/11.17 ms。该结果不含视频解码、YOLO、crop、FSM，也不是端到端视频回放；进入候选比较时仍需按上面的本地视频回放协议重测。

候选进入下一阶段需同时满足：

- 两个目标 challenge 场景的短时错误切换次数至少下降 20%，或错误状态持续时间 P90 至少下降 20%；
- 汇总四分类 event F1@0.5 和 time-IoU macro 相对最佳可比 baseline 均不下降超过 2 个百分点；本轮重点类别 `eating`、`active` 的 event recall 不出现超过 5 个百分点的回退；
- 本地同机、同后端、同 batch=1 回放时 p95 时延相对最佳可比 baseline 增长不超过 15%；在预设离线采样率（默认 4 FPS）下，回放处理不持续积压。该项只作本地环境的相对成本门槛，不声称满足摄像头实时性，也不外推为部署机器性能结论。

这些阈值是本实验的工程验收门槛，不是论文结论。若 challenge 改善但整体精度越界，保留为未通过候选并检查类别定义/转移规则，不用调 FSM 把失败指标藏起来。

## 执行顺序与验收产物

1. **数据/权重审计**：产出 manifest、哈希去重与 split 泄漏报告、类别索引映射、两个权重的 strict-load 结果；任一不通过则停止。
2. **评估器验证**：用合成特征或固定小序列验证 `window`、`stream` 张量输出与状态重置；确认两个旧 checkpoint 的类别概率非 NaN，逐帧 trace 时间戳严格递增。
3. **建立统一 baseline**：运行 I0–I3；固化 detector、预处理、FSM 参数、输出 trace、汇总 JSON/CSV、时间线图与运行环境。
4. **训练 T0/T1/T2**：固定配置和随机种子；每次训练保存配置、代码版本标识、数据 manifest 哈希、类别顺序、随机种子、最佳 epoch、checkpoint 与 sidecar。
5. **验证与选择**：只在 val/challenge 集上选择 checkpoint、重置间隔和 FSM 参数；将每个模型 raw 输出与最终状态并排查看。
6. **冻结测试**：锁定选择后，在 30 条已标注长视频上对获选方案和两个 checkpoint baselines 各运行一次；生成逐视频指标和总表。随后在本地用 dashboard 行为推理的隔离副本回放同一批视频，核对统一评估器与实际逻辑的状态转移差异；本阶段不要求接摄像头。

最终交付至少包括：可复现实验配置、数据审计报告、每个候选 checkpoint、原始逐时刻 trace、逐视频指标、总体对照 CSV、challenge 时间线图、推理耗时表，以及明确的通过/未通过结论。不得只交总体准确率或单张示例图。




## 已执行结果（2026-10-24）

所有 I0/I1/I2/I3 基线均在同一组 30 条带时间标注长视频、0.5 秒采样上运行，类别顺序为 `resting, active, eating, drinking`。逐视频 trace、指标、时间线图保存在 `experiments/rabbit_streaming_memory/outputs/`，汇总表为 `outputs/baseline_all_summary.json`。

| 对照 | time-IoU macro | event F1@0.5 | transition F1 | 说明 |
|---|---:|---:|---:|---|
| Oct20 I0 独立 16 帧窗 | 0.66712 | 0.52611 | 0.02667 | 窗口基线 |
| Oct20 I1 连续 GRU、无 FSM | 0.68426 | 0.55106 | 0.06667 | 相较 I0 三项均提升 |
| Oct20 I2 I1 + 简化 debounce-only FSM | 0.67383 | 0.42593 | 0.03750 | 此项只含 debounce/合法转移逻辑，不含 dashboard 的规则候选与置信度融合；因此是 FSM-only 消融，不是完整线上 FusionAgent |
| Oct20 I1 + `reset_after=2s` | 0.68426 | 0.55106 | 0.06667 | 3128 条记录中 reset 触发 0 次，与 I1 完全相同；本数据回放没有超过 2 秒的连续采样间隔 |
| Oct20 I3 人为每 8 条记录清空 hidden | 0.65854 | 0.47222 | 0.07820 | 3128 条记录触发 363 次清空；相较 I1 的整体 IoU/event F1 变差，transition F1 小幅上升。人为约 4 秒清空不是推荐配置，只是敏感性对照 |
| Jun24 I0 独立 16 帧窗 | 0.52616 | 0.40534 | 0.03222 | 窗口基线 |
| Jun24 I1 连续 GRU、无 FSM | 0.50307 | 0.40833 | 0.06111 | transition F1 上升，IoU 略降，event F1 基本持平 |

计时复核：旧版视频回放每次 GPU 推理前后未显式同步 CUDA，产生的 `p95_action_ms_mean_per_video` 受异步执行影响，且测量范围只包含模型循环、不含视频解码/检测；不得用于 I0/I1 时延优劣判断。隔离回放计时现已补上 CUDA 同步，后续重跑可得到有效逐步模型调用时延。当前可直接引用的公平模型-only 对照来自 `benchmark_model.py`：2026-10-24，RTX 4090、Windows 11（10.0.22631）、PyTorch 2.7.0+cu118、batch=1、224×224、100 warm-up/1000 iterations。Oct20 stream p50/p95=7.405/10.246 ms，16 帧窗口=6.935/9.598 ms；Jun24 stream=6.820/8.973 ms，16 帧窗口=6.867/8.922 ms。窗口耗时是一次完整 16 帧模型调用，不能直接和单帧 stream 单步延迟等同为同等吞吐量；该基准不含视频解码、YOLO、crop 或 FSM。

训练数据按类别目录组织，目录标签作为整段视频的逐帧训练目标，不含视频内部片段级边界。损坏视频和跨 split 完全重复文件已按授权删除，并通过新的全量哈希审计；数据变更明细见 experiments/rabbit_streaming_memory/manifests/dataset_changes_2026-10-24.json。所有实验脚本、模型副本、配置、训练权重与评估产物都在隔离实验目录，生产训练和 dashboard 代码未修改。



## 完整数据哈希审计与报告产物

初次审计 manifests/audit_full_2026-10-24.json 留作变更前证据。删除损坏训练文件及重复 validation 副本后，最终审计 manifests/audit_final_2026-10-24.json 通过：train 1040 个、val 257 个、不可读文件 0、跨 split 字节级重复 0；Oct20/Jun24 checkpoint 均 strict-load 成功。最终 train 类别数为 active 351、drinking 31、eating 244、resting 414；val 为 active 88、drinking 9、eating 61、resting 99。



`outputs/comparison_overall.csv` 包含 7 个对照的总体质量指标；`outputs/comparison_per_video.csv` 含 210 行逐视频指标；`outputs/comparison_artifact_audit.json` 记录每个对照均有 30 份 trace、30 份 metrics JSON、30 张 GT/系统时间线图。`outputs/baseline_all_summary.json` 是便于机器读取的总表。旧回放中记录的逐视频 p95 保留作历史诊断字段，但均标为未验证，不应作为通过门槛。


## 配对统计补充

`outputs/paired_bootstrap_2026-10-24.json` 对 30 条视频做了 20,000 次有放回配对 bootstrap，报告视频级差值的 95% 百分位区间；这是描述性统计，视频之间可能并非独立。Oct20 I1 相对 I0 的 time-IoU 平均差为 +0.0171，95% 区间 [-0.0245, +0.0720]；event F1@0.5 平均差为 +0.0250，区间 [-0.0278, +0.0826]；有 transition 的 15 条视频上 transition F1 平均差为 +0.0444，区间 [0, +0.1333]。Jun24 I1 相对 I0 的 time-IoU 平均差为 -0.0231，区间 [-0.0545, +0.0025]；event F1 平均差为 +0.0030，区间 [-0.0303, +0.0340]。因此现阶段结论是“连续 hidden 对 Oct20 有方向一致的总体收益，对 Jun24 主要改善转移指标但总体 IoU 不稳定”，还不能写成模型普遍提升。

## 数据清理和授权

用户授权后，删除了 7 个 258-byte 且 SHA-256 完全相同的损坏训练视频，以及 7 个与 train 逐字节重复的 val 副本；每个重复对都保留 train 文件。删除前后分别留下审计证据，变更明细在 manifests/dataset_changes_2026-10-24.json，最后以全量哈希审计确认无损坏视频、无 split 泄漏。训练选择清单为 manifests/training_selection_approved.json，记录最终 audit 哈希和实际类别计数。



## 本地时延五轮复测

`outputs/local_latency_verified_2026-10-24_5repeat_summary.json` 汇总 5 轮独立基准，每轮 100 次预热、1000 次 CUDA 同步迭代。取各轮 p95 的中位数与范围：Oct20 单帧 stream 10.246 ms [9.066, 11.077]，16 帧窗口 9.598 ms [8.556, 11.458]；Jun24 单帧 stream 8.987 ms [8.439, 17.449]，16 帧窗口 8.922 ms [8.789, 10.662]。Jun24 stream 有一轮明显长尾，故当前环境重复性一般，应以中位数和范围报告，不将单次 p95 当稳定常数。

## 训练入口核验修正

在完整审阅训练执行路径时，T0 已修正为真正的独立 16 帧窗口 + 末帧 loss；T1/T2 使用连续 12 秒 block、前 5 秒 burn-in。验证集按四类最多 12 条均衡抽样，最终每个模型验证 45 条（drinking 验证样本只有 9 条）。


## 现有挑战标注覆盖度

`outputs/annotation_challenge_audit.json` 检查了 30 条评测视频的 38 个粗粒度片段。仅有 7 条视频含多个状态片段：EATING→ACTIVE 1 次、ACTIVE→EATING 4 次、ACTIVE→RESTING 1 次，其余另有少量饮水转移。当前标注没有“低头咬草/抬头咀嚼”子状态，也没有“active 中暂停几秒后继续跑”的内部边界，所以不能把这些具体动作做成独立挑战集指标；目前只能用 eating/active/resting 的粗粒度转移和整体时序稳定性作代理。连续 12 秒单类别训练视频仍可训练动作内部状态不变的目标，但不能替代细粒度挑战标注。


## eating / active 事件指标

`outputs/event_metrics_by_class.csv` 为按类别汇总 TP/FP/FN 后计算的事件 precision/recall/F1（参考生成脚本 `src/aggregate_event_metrics.py`）。Oct20 I1→简化 I2 FSM-only：ACTIVE event F1 0.391→0.410、recall 0.600→0.533；EATING event F1 0.400→0.538、recall均为0.538；RESTING event F1 0.452→0.280。该简化 FSM-only 对 eating 的事件精度有帮助，但 active recall 和 resting precision/整体指标有代价，需要在独立验证标注上按业务权重选择，不能根据总体 F1 一刀切。DRINKING 仍属于四分类整体计算，但不作为本次关注问题的决策依据。


## 训练视频时长与短视频处理

`outputs/duration_distribution.json` 显示可读训练视频 1040 条中有 494 条短于 12 秒、19 条短于 2 秒；验证集 264 条中有 118 条短于 12 秒、5 条短于 2 秒。因此数据加载器已改为：流式 T1/T2 对短视频读取整段并把 burn-in 限制到最多 `T-1` 帧，保证至少保留一个有监督时刻；T0 对不足 16 帧的视频重复首帧左填充到 16 帧。短序列长度逻辑有单元测试，避免按固定 12 秒读取而在大量视频上解码越界失败。具体各类分布见时长报告。


## 九组训练运行器

`experiments/rabbit_streaming_memory/run_training_matrix.ps1` 已加入 T0/T1/T2 × seeds 42/43/44 的顺序运行，逐次保存日志，任一训练失败即停止，完成后生成 `training_matrix_summary.csv`。运行器只调用隔离训练脚本，未触及生产训练或线上推理代码。九次训练已完成，每次 8 epoch；outputs/checkpoints/training_matrix_summary.csv 记录验证结果。按验证 loss 选择 T0 seed44（epoch 2，loss 2.0523）、T1 seed42（epoch 7，loss 1.7276）、T2 seed43（epoch 7，loss 2.1926）。


## 数据源帧率统一

全量审计发现视频原始 FPS 混有 8、10、16、30（少量 16/30 FPS 小数值），而实验配置定义为 8 FPS。此前隔离数据加载器按原生逐帧读取，却把块长度按目标 8 FPS 解释，会让 30 FPS 视频的“12 秒块”实际只覆盖约 3.2 秒。现已修正：随机窗口长度按视频时长换算成目标帧数，读取时按 `round(target_index × source_fps / target_fps)` 从原视频抽帧；对当前所有源 FPS≥8 的数据保持单调时间采样。新增 8/10/16/30 FPS 映射测试，确保 T0 16 帧和 T1/T2 12 秒块的真实时长一致。该修正仅作用于隔离数据加载器。


## 训练 checkpoint 冻结评估（2026-10-24）

同一组 30 条带标注长视频、0.5 秒采样上，对按训练验证 loss 选出的三个 checkpoint 完成 raw 推理；T0 同时评估独立窗口和连续 hidden。三个模型又分别应用同一份固定 FSM 参数做后处理对照。

| 候选 | 推理 | time-IoU macro | event F1@0.5 | transition F1 |
|---|---|---:|---:|---:|
| Oct20 原始 | 16 帧窗口 | 0.66712 | 0.52611 | 0.02667 |
| Oct20 原始 | 连续 hidden | 0.68426 | 0.55106 | 0.06667 |
| T0 seed44 | 16 帧窗口 | 0.38229 | 0.28889 | 0.01709 |
| T0 seed44 | 连续 hidden | 0.35251 | 0.29444 | 0.00000 |
| T1 seed42 | 连续 hidden | 0.29866 | 0.17025 | 0.06965 |
| T2 seed43 | 连续 hidden | 0.24518 | 0.22407 | 0.00000 |

本轮 T0/T1/T2 微调 checkpoint 均未达到原始 Oct20 基线，且不满足计划中的总体质量门槛，因此本轮训练候选全部判定未通过，不应替换现有模型。逐类指标见 outputs/trained_vs_baselines_by_class.csv：T0 seed44 的窗口和 stream 测试回放对 13 个 eating 事件都没有检出（eating recall=0），逐时刻 raw 输出中 eating 占比为 0%；T1 eating recall 0.154，T2 为 0。FSM 后处理没有恢复 eating 检出，且整体 event F1 更低。T0 在 active 上的 recall 为 0.600（窗口）/0.667（stream），但整体质量仍显著低于 baseline。现有结果不能单独确定原因；后续应先检查逐类 logits/混淆、验证抽样与损失曲线，再决定训练策略，不应用 FSM 掩盖类别塌缩。

逐视频配对 bootstrap（20,000 次）见 outputs/trained_checkpoint_paired_bootstrap_2026-10-24.json。相较 Oct20 I1，三个训练候选 time-IoU 的 reference-minus-candidate 平均差分别为 +0.332（T0 stream）、+0.386（T1）、+0.439（T2）；区间均高于 0，支持当前已测 checkpoint 明显低于 baseline 的判断。该区间是描述性重采样结果，评测视频数量有限且可能相关。

本地同步模型-only 时延见 outputs/trained_checkpoints_latency_2026-10-24.json（RTX 4090，100 warm-up、500 iterations）。三个训练 checkpoint 单帧 stream p95 分别为 9.389 ms、9.965 ms、11.094 ms；16 帧调用 p95 分别为 10.615 ms、10.159 ms、11.015 ms。未含解码/YOLO/crop；相对计算开销接近原模型，但质量未通过。


## 全九个训练 seed 的验证混淆审计

为确认 seed 选择和类别塌缩问题，对全部 9 个已保存 checkpoint 在同一组均衡 45 条验证视频上重新计算了 endpoint confusion matrix，结果见 outputs/all_seeds_validation_confusion_2026-10-24/validation_confusion.json。三个 T0 seed 的 eating recall 均为 0；T1 seeds 的 eating recall 分别为 0.500、0、0.167；T2 seeds 分别为 0.167、0、0。所有 9 个 seed 对 drinking 的验证 recall 都为 0。T1 seed42 的验证 eating recall 最高，但在冻结长视频测试上的 eating recall 只有 0.154，显示该小验证集选择结果没有稳定泛化。按预先配置的验证 loss 选出的 T0/T1/T2 checkpoint 在 eating 上依然失败，因此不能只依据总体 val loss 选模型；后续实验应在训练前明确关键类别约束或宏平均准则，并扩大/补充验证样本。


## Dashboard FusionAgent 隔离逻辑回放

根据 dashboard 源文件 backend/ml/decision/fusion_agent.py（源码及 rule/config SHA-256、dashboard Git HEAD 记录在 manifests/dashboard_source_manifest.json；replay manifest 也记录 fusion 文件 hash），在实验目录实现了仅包含 Rabbit 运动规则、候选门限、模型调用节流、2 秒 GRU-reset bookkeeping 和 FSM 的独立快照。它直接使用已经存档的 2 FPS 概率和 sampled bbox，不调用或修改 dashboard 项目。当前线上默认 detect_fps 为 4 FPS（backend/app/domain/cameras.py），本回放只有 2 FPS 点，因此按 bbox 估出的速度历史、FSM threshold scaling 与线上配置并不一致；并且原概率由本实验 temporal-window crop 生成，不是 dashboard 的单帧 bbox+25% context crop。reset 后的模型概率也没有重新计算。故这是一项融合逻辑差异分析，不是端到端 dashboard 复现，不作为线上 KPI 或通过门槛。

| 概率来源 | 结果解码 | time-IoU macro | event F1@0.5 | transition F1 |
|---|---|---:|---:|---:|
| Oct20 I1 raw | 直接 argmax | 0.68426 | 0.55106 | 0.06667 |
| Oct20 I1 概率 | dashboard FusionAgent 快照 | 0.57412 | 0.36221 | 0.01111 |
| T0 seed44 概率 | dashboard FusionAgent 快照 | 0.37164 | 0.20833 | 0.00000 |
| T1 seed42 概率 | dashboard FusionAgent 快照 | 0.31133 | 0.23809 | 0.01212 |
| T2 seed43 概率 | dashboard FusionAgent 快照 | 0.25813 | 0.17333 | 0.01681 |

Oct20 快照输出 eating recall 为 0.385、active recall 为 0.267；三个训练 checkpoint 经此候选融合后 eating recall 均为 0。与 I2 简化 debounce-only 消融相比，这一结果展示 feeding confidence gate、active gate、box-speed rule 和采样 cadence scaling 会共同改变状态结果；但受 2 FPS trace 和不同裁剪影响，不可把此差值当作完整生产环境的因果效应，不能据此判断完整线上实现相对 raw baseline 的实际效果。输出和逐视频时间线在 outputs/dashboard_fusion_*，汇总见 dashboard_fusion_comparison.csv 与 dashboard_fusion_by_class.csv。


## 4 Hz Dashboard 风格真模型回放（2026-10-24）

为缩小此前 2 Hz 融合逻辑回放与 dashboard 的差异，在隔离目录新增 [dashboard_style_replay.py](../experiments/rabbit_streaming_memory/src/dashboard_style_replay.py)，按 4 FPS 从 30 条已标注长视频取帧，使用真实 RepViT-GRU checkpoint 逐帧预测。bbox 外扩、RGB/224 resize 和 ImageNet normalization 参照 dashboard recognizer；隔离 FusionAgent 快照按 bbox 运动规则节流模型调用、跨超过 2 秒间隔重置 GRU hidden，并执行候选门限与 FSM。每个视频都独立重置状态。脚本先对每个时刻运行一次 YOLO，再将同一检测框输入五个模型，保证组间输入相同。

运行与汇总命令：

```powershell
.venv-vlm\Scripts\python.exe experiments/rabbit_streaming_memory/src/dashboard_style_replay.py --device cuda --output-root experiments/rabbit_streaming_memory/outputs/dashboard_style_4hz
.venv-vlm\Scripts\python.exe experiments/rabbit_streaming_memory/src/score_baseline.py --run-dir experiments/rabbit_streaming_memory/outputs/dashboard_style_4hz/oct20
# 对 jun24、t0_seed44、t1_seed42、t2_seed43 各运行一次同样的评分命令
.venv-vlm\Scripts\python.exe experiments/rabbit_streaming_memory/src/summarize_dashboard_style.py
```

| 模型 | time-IoU macro | event F1@0.5 | transition F1 | eating event recall | 平均逐视频 p95 模型调用 |
|---|---:|---:|---:|---:|---:|
| Oct20 baseline | 0.58322 | 0.32148 | 0.01143 | 0.385 | 12.96 ms |
| Jun24 baseline | 0.51737 | 0.27037 | 0.00714 | 0.692 | 12.36 ms |
| T0 seed44 | 0.37868 | 0.20444 | 0.00000 | 0.000 | 11.80 ms |
| T1 seed42 | 0.31418 | 0.20939 | 0.00790 | 0.000 | 11.43 ms |
| T2 seed43 | 0.26603 | 0.15462 | 0.00926 | 0.000 | 10.91 ms |

每个模型均有 30 条视频、共 6,241 个检测到的 4 Hz 时间点和 6,211 次模型调用。RTX 4090 本地环境上，样本级 pipeline p95（视频解码、YOLO、crop/normalize）按视频平均约 32.18 ms；实际模型调用 p95 按视频平均约 10.91–12.96 ms。这里只代表本机离线处理开销，不是摄像头端到端延迟或目标设备承诺。

汇总与逐类数据位于 `experiments/rabbit_streaming_memory/outputs/dashboard_style_4hz/dashboard_style_comparison.csv`、`dashboard_style_by_class.csv` 和 `dashboard_style_comparison.json`；逐视频 trace、指标和时间线保留在各 checkpoint 子目录。所有训练候选仍显著低于 Oct20，且 T0/T1/T2 均未检出 13 个 eating 标注事件中的任何一个，不能替换现有模型。

**解释边界：**这是更接近 dashboard 的本地对照，不是 dashboard 端到端复现。它复用了当前训练仓库的兔子 YOLO 权重和离线解码，不是 dashboard 实际检测器、实时调度器或跟踪器；假设每条视频只有一个持续 track。因而不将结果称为线上 KPI。它与此前 2 Hz temporal-window trace 的输入裁剪和采样率不同，不可把两次结果差异解释为单一变量的因果效应。LSTR、TeSTra、OnlineTAS 仍只是后续架构参考，未在本轮训练矩阵直接实现或对比；T0/T1/T2 是 RepViT-GRU 训练与损失方案消融。
