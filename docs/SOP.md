# 数据处理 → 模型训练 SOP

*English version: [`SOP.en.md`](./SOP.en.md)*

> 本文档所有脚本运行前，先执行 `conda activate AR` 激活对应虚拟环境。

## 总览

```
原始raw videos → 剪成单动作短片 → 数据处理脚本（YOLO裁剪+切片）
→ 与历史数据合并成新快照 → 改 config.yaml 指向新快照 
→ 教师模型微调 → 蒸馏学生模型
```

## 步骤 1：收集原始视频

从已部署的机构现场（例如用Anydesk登录HKBR），把录制的原始长视频下载到本地训练机，按机构/日期归档，存到：

```
dataset/raw_videos/rabbit/<机构>/records/<日期>        # 兔子
dataset/raw_videos/pet/normal/<机构>/records/<日期>    # 猫狗
```

此时的视频是完整长视频，一条视频里可能包含多种行为，**不能直接用于训练**。

![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-10-47-54-image.png)

## 步骤 2：人工剪辑分类

把步骤1收集的视频进行人工剪辑分类。

**唯一标准：一个视频片段只对应一个明确的行为。并且必须严格遵循字符串一致：猫狗类别文件夹名称是[eat, drink, other]，兔子类别文件夹名称是[resting, active, eating, drinking]。**

- 猫狗剪成的类别：`eat`（吃）/ `drink`（喝）/ `other`（其他）
- 兔子剪成的类别：`resting`（休息）/ `active`（活动）/ `eating`（吃）/ `drinking`（喝）

剪完后按类别放进对应文件夹（文件夹名必须包含类别关键字），例如：

```
sorted-0626/
  eat/
    clip_001.mp4
  drink/
    clip_002.mp4
  other/
    clip_003.mp4
```

## 步骤 3：按日期归类数据

把整理好的短片放到 `dataset/raw_videos/<机构>/` 下对应的  `sorted-` 子目录（例如 `dataset/raw_videos/rabbit/hkbr-record/sorted-0624/`）。

![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-10-45-35-image.png)

## 步骤 4：跑数据处理脚本，产出训练样本

| 脚本                             | 产出格式                               |
| ------------------------------ | ---------------------------------- |
| `utils/abnormal_crop.py`       | 每个样本一个文件夹：逐帧 JPG（16帧）+ `label.txt` |
| `utils/abnormal_crop_video.py` | 每个样本一个裁剪好的 mp4 短视频，按类别文件夹存放        |

两个脚本逻辑一致（YOLO 检测裁剪目标区域 → 按固定时长切片 → 输出到新的"日期快照"目录），区别只是**产出的物理格式不同**，两种格式都能被训练脚本直接使用，后期建议使用 `utils/abnormal_crop_video.py`，直接基于视频切片训练。

操作步骤（两个脚本一样）：

1. 打开脚本，找到顶部"配置参数"区域。

2. `INPUT_BASE_DIRS`：改成本次要处理的原始短片目录（步骤3的路径），可以填多个。

3. `OUTPUT_BASE_DIR`：**改成新的日期快照目录**，命名沿用"日期"惯例，例如从 `dataset/rabbit/0626-videos` 改成 `dataset/rabbit/0703-videos`（日期用本次处理的日期）。**不要覆盖已有的旧快照目录。**

4. 确认 `CLASS_NAMES`（以及 `abnormal_crop.py` 里的 `LABEL_MAP`）和步骤2剪辑用的类别名一致，`YOLO_MODEL_PATH` 是对应物种的检测器权重。
   
   ![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-10-29-07-image.png)

5. 直接运行脚本

6. 跑完后检查输出目录下有 `train/` 和 `val/` 两个子目录，且各类别都有样本，数量不为 0。![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-10-49-35-image.png)
   
   ![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-10-49-52-image.png)

## 步骤 5：新旧数据合并成完整快照

**关键动作**：`dataset/.../<日期>` 目录里的规则是——每个日期文件夹代表"截至这个日期的完整训练数据"，不是"只有这天新增的数据"。所以步骤4产出的新快照，需要把上一个日期快照里的历史样本也拷贝进来，合并成一份完整版本：

1. 找到当前"最新有效"的旧快照目录（同物种下日期最大的那个文件夹）。
2. 把旧快照 `train/` 和 `val/` 下的所有样本文件夹/文件，拷贝进新快照对应的 `train/` / `val/` 下（文件名自带日期和来源信息，一般不会重名冲突，直接合并拷贝即可）。
3. 合并完成后，新快照目录就是"当前有效数据"，旧的日期快照保留作历史记录，**之后不要再往旧快照里加数据**。

## 步骤 6：更新 config.yaml，指向新数据

打开根目录 `config.yaml`，改 `task_configs.normal` (8-16行) 这一块。猫狗、兔子共用同一个配置槽位，训练哪个物种就把这里指向对应物种的最新快照，：

```yaml
task_configs:
  normal:
    dataset_path: "dataset/pet/normal/2026-07-10/videos"      # 改成步骤5产出的新快照路径
    class_names: ['eat','drink','other']   # 确认类别列表和顺序对得上
    finetune:
      checkpoint_path: None      
      output_dir: "outputs/pet/{date}/teacher_finetuned_{date}"
      output_model_name: "pet_teacher_{date}_ema-videos.pth"
```

⚠️ **切换猫狗/兔子时，`dataset_path`、`output_dir`、`output_model_name` 三个字段里的 `pet` / `rabbit` 字符串都要跟着改，不能只改 `class_names`**：

- `dataset_path`：指向对应物种的最新快照，路径里带 `pet` 还是 `rabbit` 要对应物种（例如 `dataset/pet/normal/2026-07-10/videos` vs `dataset/rabbit/2026-07-03-videos`）。
- `output_dir` / `output_model_name`：同样把路径/文件名里的 `pet` / `rabbit` 改成对应物种，避免猫狗训练出的教师权重被存进兔子的目录、或反过来。

三个字段没有同步改的话，虽然训练本身能跑起来，但产出的模型会存到错的物种目录下，容易和历史模型搞混。

![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-11-35-11-image.png)

## 步骤 7：教师模型微调

```bash
conda activate AR
python train/train_finetune_normal_3.py
```

直接运行脚本，教师模型会自动保存在对应日期的路径： `outputs/pet/{date}/teacher_finetuned_{date}` 或者 `outputs/rabbit/{date}/teacher_finetuned_{date}`。

## 步骤 8：蒸馏学生模型

在跑蒸馏之前，先确认 `config.yaml` 111行 `distillation.teacher_checkpoint` 指向的是步骤7刚产出的教师权重文件路径（不改的话蒸馏会用旧教师，等于没有用这次新训练的教师）。

![](C:\Users\PPCUSER\AppData\Roaming\marktext\images\2026-07-14-11-35-59-image.png)

```bash
conda activate AR
python distill/distill_normal_3.py
```

- 同样不需要传参数，自动读 `config.yaml`。
- 训练完成后，检查 `config.yaml` 里 `task_configs.normal.distillation.output_dir` 对应目录下，是否生成了两个权重文件：`output_model_name`（学生原始权重）和对应的 `*_ema.pth`（EMA 权重）。

## 步骤 9：验收 & 交付

- 确认步骤7、步骤8的输出目录下权重文件都存在，且训练日志里验证准确率是合理的数值（不是 0 或者接近随机猜测水平）。
- **部署用的是学生模型的 EMA 权重**（步骤8产出的 `*_ema.pth`），用这个文件更新线上部署的系统。

## 完整跑通一次的检查清单

- [ ] 原始视频已下载归档到 `dataset/raw_videos/...`
- [ ] 已完成剪辑分类，按类别文件夹整理好
- [ ] 已运行 `abnormal_crop.py` 或 `abnormal_crop_video.py`（任选一种输出格式），产出新日期快照，`train/val` 下都有样本
- [ ] 已把旧快照数据合并进新快照
- [ ] `config.yaml` 的 `dataset_path` / `class_names` 已指向新快照，`output_dir` / `output_model_name` 里的 `pet`/`rabbit` 字符串已改成对应物种
- [ ] `train/train_finetune_normal_3.py` 跑完，教师权重已生成
- [ ] `config.yaml` 的 `distillation.teacher_checkpoint` 已指向新教师权重
- [ ] `distill/distill_normal_3.py` 跑完，学生权重 + EMA 权重已生成
- [ ] EMA 权重已用于更新线上部署
