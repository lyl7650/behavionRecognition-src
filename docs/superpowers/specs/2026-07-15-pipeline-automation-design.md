# 行为识别模型流水线自动化 —— 设计文档

> 目标：把 SOP 里"每步手动改 config.yaml / 改各脚本硬编码路径"的流程，收敛成**一个脚手架 + 两个连续脚本**。
> 能自动化的连续步骤一口气跑完，只在真正需要人工的节点（Step2 剪辑分类、合并后体检确认、看训练结果）停下。
> 猫狗(pet) / 兔子(rabbit) 两个物种共用同一套代码，靠"物种"参数区分。
>
> 姊妹项目 `detection-src`（检测模型）已按同样思路做过一版自动化，本设计与其保持一致的架构模式，方便接手同事在两个项目间迁移认知。

日期：2026-07-15

---

## 1. 背景与动机

现状：整条流水线（`docs/SOP.en.md`）的痛点是**参数散落 + 手改易错**：

- 两个 crop 脚本（`utils/abnormal_crop.py` / `abnormal_crop_video.py`）：路径、类别、检测模型、TARGET_CLASSES 全是**文件顶部的模块级常量**，每批手改。
- 中央 `config.yaml`：训练/蒸馏读它，每次要手改 `dataset_path`、`class_names`，还有 handover 反复警告的 **`pet`/`rabbit` 三处子串要一起改**（`dataset_path`/`output_dir`/`output_model_name`），漏一个模型就存错物种目录。
- Step8 蒸馏前要手动把 `distillation.teacher_checkpoint` 指向刚训好的 teacher，否则用了旧 teacher。
- 原始数据目录（`records/`、`sorted-<日期>/<类别>`）靠手建，类别文件夹名打错会导致样本被数据加载器**静默跳过**（handover 2.2：靠文件名/文件夹名含类别关键词识别，不含则跳过且不报错）。

这些手动步骤大多能从 **`物种 + 机构 + 日期`** 三个信息，加上"按物种固定的默认值（类别表、检测模型、路径 token）"推导出来。

**真正必须人工、会保留的节点**（不自动化）：
- Step1：下载原始长视频（人工），放进脚手架建好的 `records/<日期>`
- Step2：人工剪辑分类（这是本项目最耗时的人工环节，无法自动化）
- 脚本B 合并快照后：体检确认（人工闸门）
- Step9：看训练结果 / 验收

## 2. 目标与非目标

**目标**
- 收敛成 3 个入口：脚手架（建目录）、脚本A（crop）、脚本B（合并+训练+蒸馏）。
- 唯一每次提供的信息是 `物种 / 机构 / 日期`，交互式问答输入。
- 其余参数（类别、检测模型、TARGET_CLASSES、所有路径、config 字段）由"物种查表 + 路径推导"自动得出。
- 现有 crop 脚本保持向后兼容：单独跑行为不变。
- 训练/蒸馏脚本尽量不改（已支持 `config_path`）。

**非目标**
- 不自动化人工节点（Step2 剪辑分类、看结果）。
- 不改训练超参、模型结构、蒸馏损失逻辑。
- 不碰 `loss_type`（保持 `ce`）、不碰 `other_idx`（`ce` 分支不使用它，见 4.5）。
- 不做跨机器/云端编排，只在本地 4090 训练机跑。
- 不重写现有 SOP/handover（另写一份 `AUTOMATION.md`，旧文档保留）。

**config.yaml 写入安全约定（codex review 采纳）**
- 改前创建**带时间戳的备份**（如 `config.yaml.20260715-2130.bak`），写临时文件后**原子替换**（`os.replace`），任一步失败都打印恢复命令。防止中途崩溃留下"错物种"的半成品 config。
- **Windows 反斜杠路径写进 YAML 必须用单引号或正斜杠，绝不能用双引号字符串**。因为 `"E:\new"` 里的 `\n` 会被 YAML 解析成换行符，路径就毁了。`sanitize_path` 读回时对单引号/正斜杠都正常。

## 3. 总体架构

流程按人工节点切成**脚手架 + 两段连续脚本**：

```
Step1 下载原始长视频（人工）
  ⬇ [脚手架 scaffold_dirs.py] 建 records/<日期> + sorted-<日期>/<各类别> 空目录
你把下载的视频拖进 records/<日期>（人工，脚手架不碰视频）
Step2 ⛔人工：把长视频剪成单动作短片，丢进 sorted-<日期>/<类别>
  ⬇ [脚本A run_crop.py] YOLO 抠图切片 → 新快照 dataset_root/<日期>/videos/{train,val}
⛔人工(可选)：抽查 crop 产物
  ⬇ [脚本B run_train_distill.py]
     合并旧快照 → ⏸体检确认 → 改写 config.yaml → teacher 训练 → 自动回填 → student 蒸馏
Step9 ⛔人工：看结果 / 验收（部署用 student 的 *_ema.pth）
```

新增/改造后的文件结构（7 新增 + 2 改造）：

```
behavionRecognition-src/
  pipeline_config.py        [新增·内部模块] 物种查表 + 路径推导 + 交互式问答
  config_editor.py          [新增·内部模块] 原地改写 config.yaml + 备份
  snapshot_merge.py         [新增·内部模块] 合并旧快照 + 体检报告
  scaffold_dirs.py          [新增·入口①] 脚手架：建 records + sorted 类别目录
  run_crop.py               [新增·入口②] 脚本A：crop
  run_train_distill.py      [新增·入口③] 脚本B：合并→体检→改config→teacher→回填→student
  docs/AUTOMATION.md        [新增·文档] 使用说明
  utils/abnormal_crop_video.py   [改造] 顶部常量 → run(config) 参数，默认值=现常量
  utils/abnormal_crop.py         [改造] 同上
  train/train_finetune_normal_3.py   [不改] 已支持 config_path，import 调 train()
  distill/distill_normal_3.py        [不改] 已支持 config_path，import 调 train_distill()
  config.yaml                    [运行时被脚本B 改写，改前备份 config.yaml.bak]
```

你实际会"跑"的只有 3 个入口；`pipeline_config` / `config_editor` / `snapshot_merge` 是它们共用的逻辑库。

**环境**：全部在 `conda activate AR` 下运行（本项目无独立打标环境，不涉及检测项目那种跨 conda 环境问题）。

**不是 git 仓库**：本项目根目录无 `.git`（已核实），所以不涉及 commit，文档/脚本都只是磁盘文件。测试用项目已有的 `unittest` 惯例（`tests/` 目录）。

## 4. 组件详细设计

### 4.1 pipeline_config.py（地基）

唯一集中写死"物种规则"的地方。三部分：

**① 物种查表 `SPECIES`**

```python
SPECIES = {
  "pet": {                                    # 猫狗
    "class_names": ["eat", "drink", "other"],
    "label_map": {"eat": 0, "drink": 1, "other": 2},
    "yolo_model": r"E:\ActionRecognition\detection-src\cat_dog_detect\runs\yolo11n_scale0.6_fliplr0.42\weights\yolo11n-ssb-0710.pt",
    "target_classes": [1, 2],                 # 检测模型里 1=cat 2=dog
    "raw_root": r"dataset\raw_videos\pet\normal",   # 原始视频根（注意多一层 normal）
    "dataset_root": r"dataset\pet\normal",          # 快照根
    "path_token": "pet",                      # config 里三处子串
  },
  "rabbit": {                                 # 兔子（4 类，以 SOP/handover 为准）
    "class_names": ["resting", "active", "eating", "drinking"],
    "label_map": {"resting": 0, "active": 1, "eating": 2, "drinking": 3},
    "yolo_model": r"E:\ActionRecognition\detection-src\rabbit-detect\runs\rabbit-yolo11n-0603\weights\rabbit-yolo11n-0603.pt",
    "target_classes": [0],                    # 检测模型里 0=rabbit
    "raw_root": r"dataset\raw_videos\rabbit",       # 兔子少一层 normal（正常）
    "dataset_root": r"dataset\rabbit",
    "path_token": "rabbit",
  },
}
```

> ⚠️ **维护提醒**：两个 `yolo_model` 指向 `detection-src` 训出的检测模型，会随检测模型更新而变。crop 脚本注释里的旧路径已失效（已核实），本表用的是当前实存路径。检测模型换代时记得更新这里。实现时应加载 `.pt` 验证 `model.names`，确认 pet `target_classes=[1,2]`（cat/dog）、rabbit `[0]`（rabbit）与检测模型的类别索引一致。

两个物种的目录结构本来就不一致（兔子 `raw_root` 少一层 `normal`），用每物种独立的 `raw_root`/`dataset_root` 吸收，不做统一。

**② 路径推导**：给定 `species / site / date`，算出：
- records 目录：`<raw_root>\<site>-record\records\<date>`
- sorted 目录：`<raw_root>\<site>-record\sorted-<date>\<每个类别>`
- 快照目录：`<dataset_root>\<date>\videos`（视频格式，默认，包一层 `videos`）或 `<dataset_root>\<date>`（帧格式，不包）

**③ 交互式问答 `ask_context()`**：依次问 `物种(pet/rabbit) / 机构 / 日期`。日期默认今天，**语义 = 录制日**（约定，写进 README），返回 context 对象。

### 4.2 scaffold_dirs.py（入口①·脚手架）

**纯建空目录，不碰视频，不需要输 raw 路径。**

```
1. ask_context()  →  交互问 物种/机构/日期
2. 一次建全两组目录：
   - records:  <raw_root>\<site>-record\records\<date>\
   - sorted:   <raw_root>\<site>-record\sorted-<date>\<每个类别>\
       pet    → sorted-<date>\{eat, drink, other}
       rabbit → sorted-<date>\{resting, active, eating, drinking}
3. 打印建好的目录树，提示：
   - 把下载的长视频放进 records\<date>\（你自己放）
   - 剪辑后的短片按行为丢进 sorted 的类别子文件夹
```

关键：类别子文件夹**按物种自动建全、名字正确**，杜绝手建打错导致样本被数据加载器静默跳过。

### 4.3 run_crop.py（入口②·脚本A）

命令行开关：`--frames` 切帧格式（默认视频 clip 格式）。

```
1. ask_context()  →  交互问 物种/机构/日期
2. 校验：sorted-<date> 目录存在且各类别下有视频（防日期输错/未剪辑就跑）
3. 调 crop（import 同进程调）：
     默认      → abnormal_crop_video.run(config)   → 输出 <dataset_root>\<date>\videos\{train,val}
     --frames  → abnormal_crop.run(config)         → 输出 <dataset_root>\<date>\{train,val}
   config 按物种查表自动填：
     - INPUT_BASE_DIRS = [sorted-<date>]            （只处理本批次单目录）
     - OUTPUT_BASE_DIR = 快照目录
     - CLASS_NAMES / LABEL_MAP = 物种查表
     - YOLO_MODEL_PATH = 物种查表（已修正为实存路径）
     - TARGET_CLASSES  = 物种查表（pet [1,2] / rabbit [0]）
     - 现有 crop 超参（clip 时长、MAX_DET、输出编码、每视频上限等）也进 config
4. 打印：产出快照路径、train/val 各类别样本数，提示可人工抽查
```

关键：`YOLO_MODEL_PATH`/`TARGET_CLASSES`/`CLASS_NAMES` 全由物种推导，杜绝"用兔子检测模型抠猫狗"这类错配。

> ⚠️ **改造约束（codex review 采纳）**：
> - crop 脚本的顶部常量**不能只做"常量→参数"的简单替换**：核实发现多个辅助函数（如 val 重分配、视频处理循环）**直读模块级全局变量**。`run(config)` 必须把运行配置**显式传给这些函数**，不能在 `run()` 里临时改模块全局（否则并发/重入不安全、也难测）。
> - crop 前**断言目标快照目录不存在或为空**。现有 crop 用 `exist_ok=True` 写输出，若快照目录已有旧产物会**混进本批**，导致它不再是干净的"本批新快照"，后续合并就错了。
> - 核实：两个 crop 模块 `import` 时**无副作用**（YOLO 实例化和处理循环都在 `main()`/函数内），改造安全。

### 4.4 run_train_distill.py（入口③·脚本B）

```
1. ask_context()  →  交互问 物种/机构/日期
2. 定位本批次快照 = <dataset_root>\<date>\videos（帧格式无 videos 层）
3. 合并旧快照（见 4.4.a）
   ┌─────────────────────────────────────────────────────┐
   │ 4. ⏸ 体检报告 + 暂停确认（人工闸门）                 │
   │   打印: 新快照 train/val 各类别样本数、比旧快照净增、  │
   │         有无"名字不含类别关键词、会被数据加载器静默    │
   │         跳过"的样本、有无"同一样本名同时出现在 train   │
   │         和 val"（数据泄漏警示）、class_names 对不对    │
   │   敲回车 → 继续；Ctrl+C → 干净中止                    │
   └─────────────────────────────────────────────────────┘
5. 改写 config.yaml（config_editor.py，时间戳备份 + 原子替换）见 4.4.b
6. build_train_config() 只算一次 → 存为 train_cfg（**必须在步骤 5 改写之后**，见 4.4.c）
7. teacher 训练  →  train_finetune_normal_3.train(config_override=train_cfg)
8. 自动回填（见 4.4.c）→ 用 train_cfg 算出的真实 teacher 路径写回 distillation.teacher_checkpoint
9. student 蒸馏  →  distill_normal_3.train_distill()（读改好的 config.yaml）
10. 打印 teacher/student 产物路径，强调"部署用 student 的 *_ema.pth"
```

> 🚫 **顺序阻塞项（子代理审查采纳）**：`build_train_config()` 是**从磁盘读 config.yaml** 的（`training/config.py:74-80`）。所以**必须先改写 config.yaml（步骤 5）、再 build（步骤 6）**。若顺序反了，`train_cfg` 拿到的是改写前的旧 `dataset_path`/`class_names`/`output_dir`——**换物种时 teacher 会用错数据集、错类别数训练**，且无报错。

> 核实：`train(config_override=None, config_path=None)` 和 `train_distill(...)` 真实签名都支持传入配置，且两模块 `import` 无训练/加载副作用（`train_finetune_normal_3.py:67`、`distill_normal_3.py:57`）。

#### 4.4.a 合并语义

**关键前提**：本项目 Step4 的 crop 产出的新快照**只有本批新数据**（不像检测项目 Roboflow 导出是累积的）。所以要把旧快照的**全部**历史样本拷进来。

```
新快照（只有本批新数据）为基础
把"同物种最新旧快照"的 train/val 下全部样本拷进新快照对应 train/val
样本单位 = 整个 mp4 文件 或 整个帧样本文件夹（不拆分）
按样本名（文件名 / 文件夹名）去重，同名跳过
```

- "最新旧快照" = 同物种 `dataset_root` 下、日期早于本批、train/val 非空的最新日期目录（排除本批自己）。
  - ⚠️ **codex review 采纳**：历史目录里有旧的 `0703-videos` 这类**无年份**命名，无法无歧义解析日期时，**要求人工在体检前指定用哪个旧快照**，不靠字符串排序或文件修改时间猜测（那可能选错基准、污染整批）。
- 视频格式样本 = `train/<类别>/xxx.mp4`；帧格式样本 = `train/<样本文件夹>/`（内含 16 帧 + label.txt）。整体拷贝，不拆内部文件。
- 同名跳过（历史数据文件名带日期/来源，撞名罕见）。
- ⚠️ **双层混合布局（子代理审查采纳，已核实真实存在）**：旧快照如 `dataset\rabbit\2026-06-26\` **同时有**顶层 `train/val`（帧格式）**和** `videos\train,val`（视频格式）。合并**只读与本批同格式的那一层**（本批视频格式→读旧快照的 `videos\` 层；本批帧格式→读顶层）。另一层若存在，在体检报告里**警示**，不自动混入（混入会导致两种格式/标签体系混杂）。
- ⚠️ **帧格式 label.txt 索引陷阱（子代理审查采纳）**：`VideoDataset` 读帧样本时 **label.txt（整数索引）优先于文件夹名匹配**（`video_dataset.py:124-127`），且该整数按**当前 class_names 顺序**解释。历史 label.txt 若在类别顺序变更前生成（crop 脚本注释显示兔子 LABEL_MAP 变过），会**静默错标**。体检报告要**抽样比对 label.txt 索引 ↔ 文件夹名关键词**，不一致即警示。
- ⚠️ **磁盘计数 ≠ 训练实际用量（子代理审查采纳）**：pet 三类训练时会触发 `_apply_normal3_balance`（`video_dataset.py:165-213`），`other` 类被截断到 `1.5×max(eat,drink)`。体检报告的磁盘样本数要**附一行说明**训练时 other 会被截断，避免你在闸门处按磁盘数误判分布。
- ⚠️ **数据泄漏警示（codex review 采纳）**：合并后体检报告要额外检查**同一样本名是否同时出现在 train 和 val**（跨 split 泄漏会让验证集精度虚高）。同时**帧格式标签以 `label.txt` 为准、视频格式标签以类别目录为准**，两种来源在报告里都列出。这些只是**报告出来让你在闸门处判断**，不强制中止（保持你定的"同名跳过"决策）。

#### 4.4.b config.yaml 改写范围

只改"会变的字段"，其余一律不碰：

| 字段 | 改成 |
|------|------|
| `task_configs.normal.dataset_path` | 本批次新快照路径 |
| `task_configs.normal.class_names` | 物种查表的类别 |
| `finetune.output_dir` / `output_model_name` 里的 pet/rabbit 子串 | 物种 `path_token` |
| `task_configs.normal.distillation.output_dir` 里的 pet/rabbit 子串 | 物种 `path_token` |
| `distillation.teacher_checkpoint` | 见 4.4.c（teacher 训完回填） |

**不碰**：`loss_type`（保持 `ce`）、`other_idx`（见 4.5）、`{date}` 占位符（保持现有由 `training/config.py` 填执行当天的行为）、所有训练超参。

改写实现细节：
- 改前**时间戳备份** `config.yaml` → `config.yaml.<YYYYMMDD-HHMM>.bak`，**写临时文件后原子替换**（`os.replace`），任一步失败打印恢复命令。
- 顺手规整 `dataset_path`/`teacher_checkpoint` 的 `r"E:\..."` 怪写法为干净路径字符串（`sanitize_path` 对无前缀路径照常处理）。
- ⚠️ **写 YAML 用单引号或正斜杠，禁止双引号反斜杠**（`"E:\new"` 的 `\n` 会被解析成换行符，路径毁掉）。
- ⚠️ **子串替换后断言（子代理审查采纳）**：改完 `output_dir`/`output_model_name`/`distillation.output_dir` 后，断言这三个字段确实含目标物种 token（`pet` 或 `rabbit`）。防止 config 被手工改动后子串不再匹配、替换静默失效（模型悄悄存到错物种目录）。

#### 4.4.c teacher_checkpoint 自动回填

> 核实（codex review）：teacher 实际保存路径 = `cfg["output_dir"] / cfg["output_model_name"]`（`teacher_finetune.py:234`，其中 `output_model_name` 本身已是 EMA 文件名）。**不经过** `resolve_ema_model_name()`——那个只用于 student 蒸馏输出。所以回填路径直接用 `output_dir/output_model_name`，不要套 EMA 命名函数。

- **跨午夜安全（阻塞项修复）**：teacher 训 20 epochs 很可能跨天。config 里 `{date}` 由 `training/config.py` 在**调用时**替换成当天 `MMDD`。若"训练时算一次、训完回填时又算一次"，两次 `{date}` 可能差一天，回填路径就指向一个不存在的目录。
- **做法**：**先改写 config.yaml（4.4 步骤 5），再** `build_train_config()` 只调用一次，存为 `train_cfg`；teacher 训练用 `train(config_override=train_cfg)`，回填也复用 `train_cfg`。teacher 真实路径 = `Path(train_cfg["output_dir"]) / train_cfg["output_model_name"]`。
- 蒸馏前校验该 teacher 文件确实存在，不存在则报错中止（防止用了旧 teacher 或路径算错）。

### 4.5 other_idx / loss_type 说明（为什么不碰）

handover 警告 teacher 脚本 `other_idx=2` 硬编码、换物种会算错 loss。**核实代码 `train/train_finetune_normal_3.py:69-72` 后确认此风险在当前配置下不存在**：

```python
if cfg.get("loss_type", "ce") == "hierarchical":
    criterion = HierarchicalPriorityLoss(other_idx=2, ...)   # 只有这里用 other_idx
else:
    criterion = SimpleCELoss()                               # ce 分支不碰 other_idx
```

config.yaml 里 `loss_type: ce`，走 `SimpleCELoss`，`other_idx` 根本不被使用。所以只要保持 `ce`，无需改 `other_idx`，也无需改这段代码。脚本B 改 config 时保持 `loss_type: ce` 不动。

> ⚠️ **补充（codex review 采纳）**：以上"不碰"结论**只针对 teacher**。**student 蒸馏始终调 `_resolve_other_idx(class_names)`**（`distill_normal_3.py:41`，不受 `loss_type` 控制）。当前物种表下它是对的（pet 的 `other` 在 idx 2、rabbit 的 `resting` 在 idx 0，代码能正确识别）。但**物种表的类别名或顺序一旦变更，必须在体检报告里校验背景类位置**，否则蒸馏损失会算错。

### 4.6 现有 crop 脚本改造方式（向后兼容）

| 脚本 | 改造 | 单独跑 |
|------|------|--------|
| `utils/abnormal_crop_video.py` | 顶部模块级常量 → 包成 `run(config)`，常量变 config 字段，默认值=现常量 | ✅ 保留 `if __name__=="__main__"` 用默认值跑 |
| `utils/abnormal_crop.py` | 同上 | ✅ 同上 |

训练/蒸馏脚本**不改**：`train(config_override=None, config_path=None)` 和 `train_distill(...)` 已支持传 config，orchestrator 改写 config.yaml 后直接 import 调即可。

## 5. 错误处理

- 三个入口跑重活前**先校验**：物种/机构/日期对应目录、YOLO 模型、旧快照、config.yaml 是否存在——早失败。
- crop 前校验 sorted 目录非空；**crop 前断言目标快照目录不存在或为空**（防旧产物混入本批）；合并前校验新快照非空。
- 改 config 前**时间戳备份 + 原子替换**，失败打印恢复命令。
- 旧快照日期无法无歧义解析（如 `0703-videos` 无年份）时，要求人工指定基准快照，不猜测。
- 体检确认是唯一人工闸门，Ctrl+C 干净退出。
- teacher/student 训练沿用脚本自身异常；蒸馏前校验 teacher 文件存在。

## 6. 测试策略

本项目有 `tests/` 目录和 `unittest` 惯例，顺着它写：

- `pipeline_config.py`：路径推导 + 物种查表纯逻辑，单元测试（给定 species/site/date 断言各路径正确，含兔子少一层 normal、视频格式包 videos 层）。
- **合并逻辑** `snapshot_merge.py`：构造假样本（mp4 和帧文件夹两种），验证"旧快照全拷入、同名跳过、样本整体不拆"。
- **config 改写** `config_editor.py`：验证改写后 `dataset_path`/`class_names`/三处 pet-rabbit 子串正确、`loss_type` 未被碰、`.bak` 生成、`r"..."` 被规整。
- **静默跳过检测**：构造名字不含类别关键词的样本，断言体检报告能报出来。
- **跨 split 泄漏检测**：构造同名样本同时落在 train 和 val，断言体检报告能报出来。
- **config 改写 YAML 引号**：断言写出的反斜杠路径用单引号/正斜杠、不是双引号（`\n` 陷阱），且写回后 `sanitize_path` 能正确解析。
- **teacher 路径回填**：mock `train_cfg` 的 `output_dir`/`output_model_name`，断言回填路径 = `output_dir/output_model_name`（不套 `resolve_ema_model_name`）。
- 端到端：用一小段测试视频手动走一遍全流程验收。

## 7. 决策记录

| 决策 | 选择 |
|------|------|
| 自动化边界 | 脚手架 + 两段连续脚本（脚本A crop / 脚本B 合并+训练+蒸馏） |
| 三块划分 | 脚手架独立 / crop / 合并+训练+蒸馏 |
| 参数输入 | 交互式问答（物种/机构/日期），三入口统一；crop 用 `--frames` 切帧格式 |
| 日期 | 手输批次日期，语义=录制日，全程贯穿 raw/sorted/快照；不覆盖 config 的 `{date}` 占位符 |
| 兔子类别 | 4 类 `[resting, active, eating, drinking]`（以 SOP/handover 为准） |
| 快照命名 | `<日期>\videos\`（视频格式，默认）/ `<日期>\`（帧格式），不再用 `-videos` 后缀 |
| 脚手架 | 只建空目录（records + sorted 各类别），不搬视频，不输 raw 路径 |
| crop 多目录 | 只处理本批次单 sorted 目录 |
| crop 前置校验 | 目标快照目录必须不存在或为空（防旧产物混入） |
| 合并语义 | 旧快照全拷入新快照、样本整体不拆、同名跳过 |
| 旧快照双层布局 | 只读与本批同格式的层（视频→videos/，帧→顶层），另一层存在则体检警示 |
| 合并体检 | 报告跨 split 泄漏 + 帧/视频标签来源 + label.txt 索引与文件夹名比对 + other 类训练截断说明 |
| 旧快照选择 | 日期无法无歧义解析时要求人工指定，不猜测 |
| 步骤顺序 | **先改写 config.yaml 再 build_train_config**（build 从磁盘读，反了会用旧值训练） |
| other_idx / loss_type | teacher 不碰（`ce` 不用 other_idx，已核实）；student 蒸馏始终 `_resolve_other_idx`，类别顺序变更需体检校验 |
| config 传参 | 直接改写中央 config.yaml，时间戳备份 + 原子替换 |
| YAML 引号 | 反斜杠路径用单引号/正斜杠，禁双引号；子串替换后断言含目标 token |
| r"..." 写法 | 改写时规整为干净路径字符串 |
| teacher_checkpoint | 训练前算一次 train_cfg 复用（跨午夜安全），回填 `output_dir/output_model_name`，蒸馏前校验存在 |
| 内部模块 | 拆 3 个（pipeline_config / config_editor / snapshot_merge） |
| 检测模型路径 | 修正为实存路径（crop 脚本注释里的已失效） |
| crop 改造 | 运行配置显式传给辅助函数，不改模块全局；import 无副作用（已核实） |
| 架构 | 抽 pipeline_config 解析器 + 最小改造 crop 脚本（方案一） |
| 文档 | 另写 AUTOMATION.md，旧 SOP/handover 保留 |

## 8. 文件清单（交付物）

新增 7：`pipeline_config.py`、`config_editor.py`、`snapshot_merge.py`、`scaffold_dirs.py`、`run_crop.py`、`run_train_distill.py`、`docs/AUTOMATION.md`
改造 2：`utils/abnormal_crop_video.py`、`utils/abnormal_crop.py`
不改：`train/train_finetune_normal_3.py`、`distill/distill_normal_3.py`、`training/config.py`
