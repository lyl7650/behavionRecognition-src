# Task 4 Brief: 滑块对齐"侧边栏勾选的最低档"（替换 highestTierIdx）

这是你的需求来源，使用其中的确切代码（逐字照抄，勿改）。只改一个文件：`auto_presort/review_app.html`。改前先 `Read` 整个文件确认锚点唯一存在。

## 背景与目标

审核页现有逻辑（上一版）：打开视频时，滑块跳到"这个视频自己最高能撑到的那一档"（函数 `highestTierIdx()`）。问题：每个视频最高档不同，逐个点开滑块位置乱跳，用户觉得"没有固定默认值"。

**新需求（用户已确认）：** 打开视频时，滑块对齐到**侧边栏 Confidence tier 复选框里勾选的最低那一档**（数值最小的已勾档，"No model"/null 框不算）。这样只要勾选不变，滑块位置固定。pristine（未编辑、未 Approve）视频载入该档的段（最低档段最全）。

侧边栏复选框结构（已存在）：`.tierCheck` 输入框，`data-tier` 取值 `"0.9" "0.8" "0.7" "0.6" "0.5" "null"`。CONF_LEVELS = `[0.5,0.6,0.7,0.8,0.9]`，下标 0..4。

## 修改 1：把 `highestTierIdx` 替换为 `lowestCheckedTierIdx`

`Edit` old_string：
```
function highestTierIdx() {
  // Index into CONF_LEVELS of this video's highest non-empty tier (0.9 down to 0.5).
  // Falls back to current confLevelIdx if no segment_levels / all tiers empty.
  if (!doc || !doc.segment_levels) return confLevelIdx;
  for (let i = CONF_LEVELS.length - 1; i >= 0; i--) {
    const key = CONF_LEVELS[i].toFixed(1);
    if (doc.segment_levels[key] && doc.segment_levels[key].length > 0) return i;
  }
  return confLevelIdx;
}
```
new_string：
```
function lowestCheckedTierIdx() {
  // Index into CONF_LEVELS of the lowest-valued CHECKED sidebar tier box ("null" box excluded).
  // Returns current confLevelIdx when no numeric tier is checked (fallback, no change).
  let lowest = -1;
  for (const cb of document.querySelectorAll(".tierCheck:checked")) {
    const t = cb.dataset.tier;
    if (t === "null") continue;
    const idx = CONF_LEVELS.indexOf(parseFloat(t));
    if (idx >= 0 && (lowest < 0 || idx < lowest)) lowest = idx;
  }
  return lowest < 0 ? confLevelIdx : lowest;
}
```

## 修改 2：`loadVideo` 里改用新函数

注意：新逻辑不再依赖 `doc.segment_levels` 是否非空来决定档位（档位来自侧边栏勾选），但载入段时仍需 `doc.segment_levels` 存在且该档 key 有数据。保持"旧格式 doc（无 segment_levels）不动"的行为。

`Edit` old_string：
```
  if (doc.segment_levels) {
    // Align the slider to this video's own highest tier (kills stale global confLevelIdx).
    confLevelIdx = highestTierIdx();
    // For a pristine video (not approved, no human edits) load that tier's segments so the
    // timeline shows trims immediately instead of the possibly-empty default 0.9 tier.
    const pristine = !doc.approved && !doc.segments.some(s => s.source === "human");
    if (pristine) {
      const key = levelKey();
      if (doc.segment_levels[key]) {
        doc.segments = JSON.parse(JSON.stringify(doc.segment_levels[key]));
      }
    }
  }
```
new_string：
```
  if (doc.segment_levels) {
    // Align the slider to the lowest CHECKED sidebar tier (fixed position across videos).
    confLevelIdx = lowestCheckedTierIdx();
    // For a pristine video (not approved, no human edits) load that tier's segments so the
    // timeline shows trims immediately instead of the possibly-empty default 0.9 tier.
    const pristine = !doc.approved && !doc.segments.some(s => s.source === "human");
    if (pristine) {
      const key = levelKey();
      if (doc.segment_levels[key]) {
        doc.segments = JSON.parse(JSON.stringify(doc.segment_levels[key]));
      }
    }
  }
```

## 验证（前端无自动化测试，做静态自检）

1. 用 `Read` 重新读改后文件，确认两处修改都在，且文件里**不再有** `highestTierIdx` 的任何引用（全部改成了 `lowestCheckedTierIdx`）。用 grep 或通读确认没有遗漏的旧函数名调用。
2. JS 语法配平（括号/大括号成对）。
3. 逐个推演行为，确认：
   - 全选（0.5~0.9 都勾）→ lowest = CONF_LEVELS.indexOf(0.5) = 0 → 滑块停 0.5 → pristine 视频载入 "0.5" 档段（最全）。
   - 只勾 0.8、0.9 → lowest = indexOf(0.8) = 3 → 滑块停 0.8。
   - 只勾 "No model"（null）或一个数值档都没勾 → 循环里没有有效 idx → 返回原 confLevelIdx（不变，不崩）。
   - 已编辑/已批准视频（pristine=false）→ 只对齐滑块，不覆盖 doc.segments。
   - 旧格式 doc（无 segment_levels）→ 整个 if 跳过，行为不变。
4. 确认没有引入未定义符号：`CONF_LEVELS`、`confLevelIdx`、`doc`、`levelKey` 均已定义；`document.querySelectorAll` 正常。
5. 确认无新中文/emoji（本次新代码注释均英文）。

不要跑 pytest。不要改其他文件。

## 硬约束
- 本仓库**不是 git 仓库**——不要执行 git 命令，不要 commit。
- 只改 `auto_presort/review_app.html`。
- 保持既有风格（2 空格缩进、`$("id")`、`var(--xxx)`）。
- 界面/注释新增文字全英文，无中文、无 emoji。
