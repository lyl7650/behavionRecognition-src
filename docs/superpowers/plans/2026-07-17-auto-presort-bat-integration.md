# auto_presort 并入 .bat 自动化流程 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 auto_presort 的 label→review→apply 三步包装成"问物种/机构/日期即自动推路径"的入口 + 三个双击 .bat，无缝插进现有 1_scaffold → 2_run_crop → 3_train 流程。

**Architecture:** 三个薄 wrapper（`run_presort_label/review/apply.py`）复用 `pipeline_config.ask_context` 收集 物种/机构/日期，用现有路径推导函数（`records_dir`/`sorted_dir`）算出路径，拼成 argv 交给现有 `auto_presort_*.py` 的 `main(argv)` 执行——不复制任何业务逻辑。teacher 路径按物种从 `SPECIES` 取默认、运行时可粘贴覆盖。

**Tech Stack:** Python 3.12（conda `AR`）；stdlib（argparse）；现有 `pipeline_config`、`auto_presort_*` CLI；Windows .bat。

**运行环境：** 测试一律 `conda run -n AR python -m pytest <path> -v`（项目根目录）。**不是 git 仓库**：无法 commit，每个 Task 完成 = 对应测试全绿（沿用现有约定）。wrapper 的 `main()` 里调用 torch/YOLO 的部分**不写单测**（既定风格），只单测纯路径推导函数。入口脚本 print 一律英文（cp1252 约定）。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| Modify `pipeline_config.py` | `SPECIES[pet/rabbit]` 各加 `teacher_checkpoint` 默认路径字段 |
| Create `run_presort_label.py` | 纯函数 `build_label_argv(ctx, checkpoint)` + `resolve_checkpoint(species, input_fn)` + `main()` |
| Create `run_presort_review.py` | 纯函数 `build_review_argv(ctx)` + `main()` |
| Create `run_presort_apply.py` | 纯函数 `build_apply_argv(ctx)` + `main()` |
| Create `2a_presort_label.bat` / `2b_presort_review.bat` / `2c_presort_apply.bat` | 双击启动器 |
| Modify `docs/AUTOMATION.md` | 把三步作为标准剪辑步骤写入 + 顺序图 + 人工节点更新 |
| Create `tests/test_run_presort.py` | 纯路径推导 + checkpoint 解析的单测 |

设计要点：每个 wrapper 把"推导路径"抽成**无副作用的纯函数**（返回 argv 列表 / 解析后的路径字符串），`main()` 只负责 `ask_context` + 调纯函数 + 委派给现有 `main(argv)`。纯函数可脱离 torch/YOLO 单测。

---

## Task 1: SPECIES 增加 teacher_checkpoint 默认路径

**Files:**
- Modify: `pipeline_config.py:11-28`（`SPECIES` 的 `pet` 与 `rabbit` 块）
- Test: `tests/test_pipeline_config.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_pipeline_config.py` 末尾：

```python
def test_species_has_teacher_checkpoint_defaults():
    pet = pc.SPECIES["pet"]["teacher_checkpoint"]
    rabbit = pc.SPECIES["rabbit"]["teacher_checkpoint"]
    assert pet.endswith("pet_teacher_0710_ema-videos.pth")
    assert "outputs" in pet and "pet" in pet
    assert rabbit.endswith("rabbit_teacher_0706_ema-videos.pth")
    assert "outputs" in rabbit and "rabbit" in rabbit
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_pipeline_config.py::test_species_has_teacher_checkpoint_defaults -v`
Expected: FAIL with `KeyError: 'teacher_checkpoint'`

- [ ] **Step 3: 加字段**

在 `pipeline_config.py` 的 `SPECIES["pet"]` 块内（`path_token` 之后）加一行：

```python
        "teacher_checkpoint": r"E:\ActionRecognition\behavionRecognition-src\outputs\pet\2026-07-10\teacher_finetuned_3cls_0710\pet_teacher_0710_ema-videos.pth",
```

在 `SPECIES["rabbit"]` 块内（`path_token` 之后）加一行：

```python
        "teacher_checkpoint": r"E:\ActionRecognition\behavionRecognition-src\outputs\rabbit\2026-07-06\teacher_finetuned_3cls_0706\rabbit_teacher_0706_ema-videos.pth",
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_pipeline_config.py -v`
Expected: PASS（含新增用例，原有用例不回归）

---

## Task 2: run_presort_label.py（含 teacher 默认/覆盖）

**Files:**
- Create: `run_presort_label.py`
- Test: `tests/test_run_presort.py`（新建）

- [ ] **Step 1: 写失败测试**

新建 `tests/test_run_presort.py`：

```python
import pipeline_config as pc
import run_presort_label as lbl


def test_resolve_checkpoint_uses_default_on_blank():
    got = lbl.resolve_checkpoint("pet", input_fn=lambda p: "")
    assert got == pc.SPECIES["pet"]["teacher_checkpoint"]


def test_resolve_checkpoint_uses_pasted_override():
    got = lbl.resolve_checkpoint("rabbit", input_fn=lambda p: r"D:\tmp\new.pth")
    assert got == r"D:\tmp\new.pth"


def test_resolve_checkpoint_strips_surrounding_quotes():
    # Windows "复制文件地址" 粘贴出来带双引号，必须剥掉否则 isfile 失败
    got = lbl.resolve_checkpoint("pet", input_fn=lambda p: '"D:\\tmp\\new.pth"')
    assert got == "D:\\tmp\\new.pth"


def test_build_label_argv_derives_records_dir():
    ctx = {"species": "rabbit", "site": "hkbr", "date": "2026-07-15"}
    argv = lbl.build_label_argv(ctx, r"D:\tmp\new.pth")
    expected_dir = pc.records_dir("rabbit", "hkbr", "2026-07-15")
    assert argv == ["--species", "rabbit", "--dir", expected_dir,
                    "--checkpoint", r"D:\tmp\new.pth"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_run_presort.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_presort_label'`

- [ ] **Step 3: 写实现**

新建 `run_presort_label.py`：

```python
"""问答式包装：物种/机构/日期 + teacher → 调 auto_presort_label。

records_dir（长视频下载区）自动作为 --dir；teacher 默认按物种取，回车用默认或粘贴覆盖。
纯路径推导抽成 build_label_argv/resolve_checkpoint 便于单测；main() 委派给现有 CLI。
"""
import sys
import pipeline_config as pc


def resolve_checkpoint(species, input_fn=None):
    """回车用 SPECIES 默认；粘贴则用粘贴的路径（剥掉 Windows 复制路径常带的双引号）。"""
    if input_fn is None:
        input_fn = input
    default = pc.SPECIES[species]["teacher_checkpoint"]
    ans = input_fn(f"Teacher checkpoint [default {default}]: ").strip().strip('"')
    return ans if ans else default


def build_label_argv(ctx, checkpoint):
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    return ["--species", ctx["species"], "--dir", records,
            "--checkpoint", checkpoint]


def main():
    ctx = pc.ask_context()
    checkpoint = resolve_checkpoint(ctx["species"])
    argv = build_label_argv(ctx, checkpoint)
    import auto_presort_label
    return auto_presort_label.main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_run_presort.py -v`
Expected: PASS（4 个用例）

---

## Task 3: run_presort_review.py

**Files:**
- Create: `run_presort_review.py`
- Test: `tests/test_run_presort.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_run_presort.py`（`import run_presort_review as rvw` 归拢到文件顶部的 import 区，与现有测试风格一致；函数追加到末尾）：

```python
def test_build_review_argv_derives_records_dir():
    ctx = {"species": "pet", "site": "fotan", "date": "2026-07-15"}
    argv = rvw.build_review_argv(ctx)
    expected_dir = pc.records_dir("pet", "fotan", "2026-07-15")
    assert argv == ["--dir", expected_dir]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_run_presort.py::test_build_review_argv_derives_records_dir -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_presort_review'`

- [ ] **Step 3: 写实现**

新建 `run_presort_review.py`：

```python
"""问答式包装：物种/机构/日期 → 对同一 records_dir 开审核服务器。"""
import sys
import pipeline_config as pc


def build_review_argv(ctx):
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    return ["--dir", records]


def main():
    ctx = pc.ask_context()
    argv = build_review_argv(ctx)
    import auto_presort_review
    return auto_presort_review.main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_run_presort.py -v`
Expected: PASS（5 个用例）

---

## Task 4: run_presort_apply.py

**Files:**
- Create: `run_presort_apply.py`
- Test: `tests/test_run_presort.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_run_presort.py`（`import run_presort_apply as apl` 归拢到文件顶部的 import 区；函数追加到末尾）：

```python
def test_build_apply_argv_derives_in_and_out_dirs():
    ctx = {"species": "rabbit", "site": "hkbr", "date": "2026-07-15"}
    argv = apl.build_apply_argv(ctx)
    in_dir = pc.records_dir("rabbit", "hkbr", "2026-07-15")
    out_dir = pc.sorted_dir("rabbit", "hkbr", "2026-07-15")
    assert argv == ["--dir", in_dir, "--out", out_dir]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n AR python -m pytest tests/test_run_presort.py::test_build_apply_argv_derives_in_and_out_dirs -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_presort_apply'`

- [ ] **Step 3: 写实现**

新建 `run_presort_apply.py`：

```python
"""问答式包装：物种/机构/日期 → records_dir 作输入、sorted_dir 作输出，切片落地。

sorted_dir 正是 2_run_crop 读取的输入目录，故 apply 完可直接接 run_crop。
"""
import sys
import pipeline_config as pc


def build_apply_argv(ctx):
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    sorted_out = pc.sorted_dir(ctx["species"], ctx["site"], ctx["date"])
    return ["--dir", records, "--out", sorted_out]


def main():
    ctx = pc.ask_context()
    argv = build_apply_argv(ctx)
    import auto_presort_apply
    return auto_presort_apply.main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n AR python -m pytest tests/test_run_presort.py -v`
Expected: PASS（6 个用例）

---

## Task 5: 三个双击 .bat

**Files:**
- Create: `2a_presort_label.bat`
- Create: `2b_presort_review.bat`
- Create: `2c_presort_apply.bat`

无自动化测试（.bat 无法在 conda run 下单测）；验收 = 内容正确 + 手动双击冒烟（人工节点）。照搬现有 `2_run_crop.bat` 模板（`cd /d "%~dp0"` + `call conda activate AR` + `pause`）。

- [ ] **Step 1: 写 `2a_presort_label.bat`**

```bat
@echo off
REM ============================================================
REM  Step 2a: auto_presort LABEL (run_presort_label.py)
REM  Model pre-labels long videos. Needs GPU. Double-click to run.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo Running auto_presort LABEL...
python run_presort_label.py

echo.
echo ============================================================
if errorlevel 1 (
    echo  [ERROR] Label step failed. Read the messages above.
    echo  Do NOT continue to 2b until this succeeds.
) else (
    echo  Done. Next: double-click 2b_presort_review.bat
)
echo ============================================================
pause
```

- [ ] **Step 2: 写 `2b_presort_review.bat`**

```bat
@echo off
REM ============================================================
REM  Step 2b: auto_presort REVIEW (run_presort_review.py)
REM  Opens the browser review page. No GPU needed.
REM  When finished reviewing, stop the server (Ctrl+C) or just
REM  close this window; then double-click 2c_presort_apply.bat.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo ============================================================
echo  Review server is starting. A browser tab will open.
echo  Approve each video, then press Ctrl+C (or close window).
echo  AFTER that, double-click 2c_presort_apply.bat
echo ============================================================
echo.
python run_presort_review.py

echo.
echo  Server stopped. Next: double-click 2c_presort_apply.bat
pause
```

> 说明：Windows 下 Ctrl+C 会让 cmd 弹 `Terminate batch job (Y/N)?`，选 Y 后尾部 echo/pause 可能被跳过——所以"下一步是 2c"的提示在**启动前**已完整打印过，尾部只是 best-effort。

- [ ] **Step 3: 写 `2c_presort_apply.bat`**

```bat
@echo off
REM ============================================================
REM  Step 2c: auto_presort APPLY (run_presort_apply.py)
REM  Cuts approved clips into sorted-<date>/<class>/. No GPU needed.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo Running auto_presort APPLY...
python run_presort_apply.py

echo.
echo ============================================================
if errorlevel 1 (
    echo  [ERROR] Apply step failed. Read the messages above.
) else (
    echo  Done. Check any [WARN] lines above, then double-click 2_run_crop.bat
)
echo ============================================================
pause
```

> 说明：apply 的 CLI 即使有个别 clip 失败也可能以 exit 0 收尾（既有行为，不改），失败细节打在 `[WARN]` 行——所以提示用户先看 WARN 再进 `2_run_crop`。

- [ ] **Step 4: 冒烟检查（不启动重逻辑）**

Run: `conda run -n AR python -c "import run_presort_label, run_presort_review, run_presort_apply; print('IMPORT_OK')"`
Expected: 打印 `IMPORT_OK`（确认三个 wrapper 无语法/导入错误）

---

## Task 6: AUTOMATION.md 更新

**Files:**
- Modify: `docs/AUTOMATION.md`（"最省事：双击 .bat" 表格 + auto_presort 小节 + 人工节点小节）

无自动化测试（纯文档）。验收 = 内容符合 spec §3.4。

- [ ] **Step 1: 在双击 .bat 表格中插入 2a/2b/2c 三行**

把 `docs/AUTOMATION.md` 中现有表格（`1_scaffold_dirs.bat` / `2_run_crop.bat` / `3_run_train_distill.bat` 那张）替换为下面这张：在 1 和 2 之间插入三行，**同时把 `2_run_crop.bat` 的作用从"切片"改写为"裁目标区域、生成快照"**（"切片"语义现已归 2c，必须消歧义，避免两行都叫"切片"）：

```markdown
| 双击文件 | 相当于 | 作用 |
|----------|--------|------|
| `1_scaffold_dirs.bat` | `python scaffold_dirs.py` | 建目录 |
| `2a_presort_label.bat` | `python run_presort_label.py` | 模型预标长视频（需 GPU） |
| `2b_presort_review.bat` | `python run_presort_review.py` | 网页审核批准 |
| `2c_presort_apply.bat` | `python run_presort_apply.py` | 切片落地到 sorted 目录 |
| `2_run_crop.bat` | `python run_crop.py [--frames]` | 裁目标区域、生成快照 |
| `3_run_train_distill.bat` | `python run_train_distill.py [--frames]` | 合并 + 训练 + 蒸馏 |
```

- [ ] **Step 2: 把"模型辅助预剪辑（auto_presort，可选）"小节改成标准步骤 + 顺序图**

把现有 `## 模型辅助预剪辑（auto_presort，可选）` 整节（从标题到 `## 日期约定` 之前）替换为：

```markdown
## 剪辑分类：模型预标 + 网页审核（2a/2b/2c）

下载完长视频后，用微调好的 teacher 模型自动预切分预标注，人只需在网页上审核批准，替代手工剪辑分类。双击 `2a → 2b → 2c` 依次跑：

- **2a 预标注**（需 GPU）：模型把长视频切成带标签的候选片段。teacher 模型按物种自动选默认；运行时提示 `Teacher checkpoint [default ...]:`，回车用默认、或粘贴一个临时新模型路径覆盖。
- **2b 网页审核**（不需 GPU）：自动开浏览器，一个页面审全批。可拖动调整片段边界、改标签、删除、在空白处拖拽新建片段。**每个视频必须点 Approve 才会被切片；任何编辑都会让批准失效**（防止模型错标混进训练集）。如提示端口被占用，先关掉上一个还开着的审核窗口再重试。
- **2c 切片落地**（不需 GPU）：只切"已批准"的视频，输出到 `sorted-<日期>/<类别>/`，正是下一步 `2_run_crop` 读取的目录，直接接着往下走。文件名内嵌来源信息用于防 train/val 泄漏，**不要改名**。

完整顺序：

```
1_scaffold（建目录 + 下载长视频）
   → 2a_label（模型预标）→ 2b_review（网页审核批准）→ 2c_apply（切片落地）
   → 2_run_crop（裁目标区域、生成快照）
   → 3_run_train_distill（合并 + 训练 + 蒸馏）
```

置信度阈值和边界内缩目前用脚本内默认值（0.90 / 1.0 秒）；如需按批次调整，可改用命令行方式（见下）直接传 `--threshold` / `--trim-margin`。

### 命令行方式（等价，供习惯终端的人）

```bash
python auto_presort_label.py  --species rabbit --dir <长视频目录> --checkpoint <teacher.pth> [--threshold 0.90] [--trim-margin 1.0]
python auto_presort_review.py --dir <同一目录>
python auto_presort_apply.py  --dir <同一目录> --out <sorted-日期 目录>
```

手工剪辑分类仍可作为后备（把剪好的短片按类别放进 `sorted-<日期>/<类别>/`），与 2a/2b/2c 产出的目录结构一致。
```

- [ ] **Step 3: 更新"人工节点"小节第 2 条**

把 `## 人工节点` 小节里的第 2 条（现为"剪辑分类"）替换为：

```markdown
2. **网页审核批准**：双击 `2a` 让模型预标，再双击 `2b` 在浏览器里逐个视频审核、调整、点 Approve（人不再手工剪辑）。手工剪辑分类作为后备仍可用。
```

- [ ] **Step 4: 修正文档里过时的"三个"措辞（避免与 6 个 .bat 自相矛盾）**

现有 `docs/AUTOMATION.md` 有三处写死"三个"，插入 2a/2b/2c 后会自相矛盾，逐处修改：

- 第 3 行 `收敛成三个入口脚本` → `收敛成一串双击入口脚本`
- 第 12 行 `项目根目录下有三个双击启动器` → `项目根目录下有若干双击启动器`
- 第 27 行标题 `## 三个入口（手动命令行方式）` → `## 各入口（手动命令行方式）`

同时第 22 行"所以三个 `.bat` 必须和脚本放在同一目录"里的"三个"改为"这些"。

- [ ] **Step 5: 人工验收**

打开 `docs/AUTOMATION.md` 通读，确认：表格含 2a/2b/2c、`2_run_crop` 作用已改为"裁目标区域、生成快照"、顺序图为单条直线、命令行方式保留、teacher 默认/覆盖说明在位、无残留"三个入口/三个启动器"字样。

---

## Task 7: 全仓库回归

- [ ] **Step 1: 跑全量测试**

Run: `conda run -n AR python -m pytest tests/ -q`
Expected: 全绿，且新增 7 个用例（3 个 build_*_argv + 3 个 resolve_checkpoint〔default/override/去引号〕+ 1 个 pipeline_config teacher 字段）全部通过。当前基数为 140，执行完本计划应为 147 passed（不严格卡死总数——以"全绿 + 新增 7 个通过"为准，避免他人先前改动导致基数漂移时误判）。

- [ ] **Step 2: 三个 wrapper --help/import 冒烟**

Run: `conda run -n AR python -c "import run_presort_label, run_presort_review, run_presort_apply, pipeline_config; print(pipeline_config.SPECIES['pet']['teacher_checkpoint']); print('OK')"`
Expected: 打印 pet 默认 checkpoint 路径 + `OK`

---

## Self-Review（对照 spec）

- spec §3.1 SPECIES 默认 teacher（含两条确切路径）→ Task 1 ✅
- spec §3.2 三个问答式 wrapper（复用 ask_context、推 records/sorted、委派现有 main、不复制逻辑、teacher 可覆盖）→ Task 2/3/4 ✅
- spec §3.3 三个 .bat（照搬模板、2a 标注需 GPU）→ Task 5 ✅
- spec §3.4 AUTOMATION.md（标准步骤表格、顺序图、人工节点更新、保留命令行）→ Task 6 ✅
- spec §4 YAGNI：不改现有三 .bat 编号、不碰 run_train_distill、不复制逻辑、不动现有 CLI → 各 Task 均未触碰 ✅
- spec §5 成功标准：apply 输出 = run_crop 输入（sorted_dir 同一函数推导）→ Task 4 测试断言 in/out 目录一致 ✅
- 类型一致性：`ctx` 键 `species/site/date` 全程一致；纯函数名 `build_*_argv` / `resolve_checkpoint` 前后一致；无占位符。
