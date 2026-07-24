# Task 3 Brief: 修档位-滑块不联动 bug + ↑↓ 切换视频

这是你的需求来源，使用其中的确切代码（逐字照抄，勿改）。只改一个文件：`auto_presort/review_app.html`。改前先 `Read` 整个文件确认每处锚点唯一存在。

## 背景与目标

审核页有两个独立控件：
- 侧边栏 **Confidence tier** 复选框：筛左侧视频列表。
- 主区 **Confidence ≥ 滑块**：控制当前视频时间轴显示哪一档的 trim。滑块档位存在全局变量 `confLevelIdx`（0..4 对应 CONF_LEVELS `[0.5,0.6,0.7,0.8,0.9]`）。

**Bug：** 打开视频时，代码不按滑块档位重载，也不把滑块对准本视频。低把握视频（比如最高只到 0.7 档）在默认 0.9 档下时间轴是空的，用户得手动拖滑块才显示 trim；且 `confLevelIdx` 是全局的、切视频不重置，导致"滑块显示 0.6 但实际按 0.9 过滤"的错位。

**修复目标（用户已确认）：** 打开视频时，把滑块自动对准"这个视频最高能撑到的那一档"（= segment_levels 里从 0.9 往下第一个非空档）。对**未编辑、未 Approve** 的"干净"视频，同时把该档的段载入时间轴，保证 trim 立即显示、永不空。已人工编辑/已批准的视频保留其存盘段，不覆盖。

**附加需求：** ↑/↓ 键在**当前筛选后的可见列表**里切到上一个/下一个视频。（空格播放/暂停已有绑定，不动。）

---

## 修改 1：加全局变量 `shownList`

`Edit` old_string：
```
let filter = "all"; // sidebar filter: all | labeled (has model-generated segments)
```
new_string：
```
let filter = "all"; // sidebar filter: all | labeled (has model-generated segments)
let shownList = []; // rel_paths currently visible in sidebar (post-filter order), for ↑/↓ nav
```

## 修改 2：加 `highestTierIdx()` 函数

在 `function levelKey()` 之前插入。`Edit` old_string：
```
function levelKey() { return CONF_LEVELS[confLevelIdx].toFixed(1); }
```
new_string：
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

function levelKey() { return CONF_LEVELS[confLevelIdx].toFixed(1); }
```

## 修改 3：`syncConfUI` 里把滑块滑块位置也对齐

现在 `syncConfUI` 只改了 `confVal` 文字，没同步滑块 thumb 位置（`slider.value`），这正是"滑块显示和实际档位错位"的一半原因。

`Edit` old_string：
```
  const slider = $("confSlider");
  const enabled = doc && doc.segment_levels;
  slider.disabled = !enabled;
  $("conflevel").style.opacity = enabled ? "1" : "0.4";
```
new_string：
```
  const slider = $("confSlider");
  const enabled = doc && doc.segment_levels;
  slider.disabled = !enabled;
  slider.value = confLevelIdx;   // keep thumb aligned with the active tier
  $("conflevel").style.opacity = enabled ? "1" : "0.4";
```

## 修改 4：`loadVideo` 打开时对齐档位 + 干净视频载入该档段

`Edit` old_string：
```
  $("status").textContent = doc.approved ? "Approved ✅" : "";  // reflect the current video's real status
  syncConfUI();
  render(); loadList();
}
```
new_string：
```
  $("status").textContent = doc.approved ? "Approved ✅" : "";  // reflect the current video's real status
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
  syncConfUI();
  render(); loadList();
}
```

## 修改 5：`loadList` 里记录可见列表顺序（供 ↑↓ 用）

`Edit` old_string：
```
  shown = shown.filter(v => checkedTiers.includes(v.conf_tier || "null"));
  const el = $("vlist"); el.innerHTML = "";
```
new_string：
```
  shown = shown.filter(v => checkedTiers.includes(v.conf_tier || "null"));
  shownList = shown.map(v => v.rel_path);   // post-filter order for ↑/↓ navigation
  const el = $("vlist"); el.innerHTML = "";
```

## 修改 6：`onkeydown` 加 ↑/↓ 切视频

`Edit` old_string：
```
  } else if (e.key === "Backspace") {             // Backspace: delete selected segment
    e.preventDefault();
    if (doc && selected >= 0) $("deleteBtn").onclick();
  }
};
```
new_string：
```
  } else if (e.key === "Backspace") {             // Backspace: delete selected segment
    e.preventDefault();
    if (doc && selected >= 0) $("deleteBtn").onclick();
  } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {  // ↑/↓: prev/next video in filtered list
    e.preventDefault();
    if (!shownList.length) return;
    const cur = shownList.indexOf(curRel);
    let next;
    if (e.key === "ArrowDown") next = cur < 0 ? 0 : Math.min(cur + 1, shownList.length - 1);
    else next = cur < 0 ? 0 : Math.max(cur - 1, 0);
    if (shownList[next] && shownList[next] !== curRel) loadVideo(shownList[next]);
  }
};
```

## 修改 7：说明栏快捷键文字补上 ↑/↓

`Edit` old_string：
```
      Shortcuts: <kbd>Space</kbd> play/pause · <kbd>Enter</kbd> Approve · <kbd>Backspace</kbd> delete selected segment · <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo last step.
```
new_string：
```
      Shortcuts: <kbd>Space</kbd> play/pause · <kbd>↑</kbd>/<kbd>↓</kbd> previous/next video · <kbd>Enter</kbd> Approve · <kbd>Backspace</kbd> delete selected segment · <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo last step.
```

---

## 验证（前端无自动化测试，做静态自检）

1. 用 `Read` 重新读改后文件，确认 7 处修改都在。
2. JS 语法配平：修改 6 新增分支的括号/大括号成对，`onkeydown` 整个箭头函数仍以 `};` 收尾。
3. 逐个推演行为，确认：
   - **`highestTierIdx`**：doc 无 segment_levels → 返回原 confLevelIdx；segment_levels 各档都空 → 返回原 confLevelIdx；否则返回从 0.9 往下第一个非空档的下标。
   - **loadVideo 干净视频**：`!doc.approved && 没有 source==="human" 的段` → pristine=true → confLevelIdx 跳到最高档 → doc.segments 深拷贝载入该档 → syncConfUI 让滑块 thumb/文字对齐 → render 显示 trim。
   - **loadVideo 已编辑/已批准视频**：pristine=false → 只对齐 confLevelIdx 与滑块显示，**不覆盖** doc.segments（保留人工结果）。
   - **↑/↓**：在 shownList（当前筛选顺序）里移动，到顶/到底不动，当前视频不在列表时按 ↓/↑ 跳到列表第一个。
4. 确认没有引入未定义符号：`doc`、`confLevelIdx`、`CONF_LEVELS`、`levelKey`、`syncConfUI`、`curRel`、`loadVideo`、`shownList` 都已定义。
5. 确认新增界面文字全英文，无中文、无 emoji。

已知可接受的局限（无需处理，仅记录）：若用户在某干净视频上**只删除**了段（未 trim/relabel，剩余段仍是 model），本次判定仍视其为 pristine，重开时会按最高档重载、恢复被删的段。此情形罕见且视频未 Approve，风险低。

不要跑 pytest（本任务不涉及 Python）。不要改其他文件。

## 硬约束
- 本仓库**不是 git 仓库**——不要执行 git 命令，不要 commit。
- 只改 `auto_presort/review_app.html`。
- 保持既有风格（2 空格缩进、`$("id")` 取元素、`var(--xxx)` CSS 变量）。
- 界面文字全英文，无中文、无 emoji。
