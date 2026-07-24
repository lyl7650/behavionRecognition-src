# Task 4 Brief: review_app.html 加置信度滑块 + 换档逻辑 + 旧 doc 兼容

这是你的需求来源，使用其中的确切代码（逐字照抄，勿改）。只改一个文件：`auto_presort/review_app.html`。

## 背景
审核页时间轴显示模型预标注的段。doc 现在含 `segment_levels`（5 档段，key `"0.5"`…`"0.9"`），`doc.segments` 初始 = 0.9 档。要加一个 min 滑块选择显示/编辑哪一档：拖到某档就把该档段载入 `doc.segments`。默认停在 0.9（= 现状观感）。旧 doc（无 `segment_levels`）滑块置灰、段照常显示、不报错。换档前把当前段推进 undo 栈（Ctrl+Z 可撤回）。

用 `Edit` 工具做下面 6 处修改，每处的 old_string 都能在文件中唯一匹配。

## 修改 1：加滑块 HTML

`Edit` old_string：
```
  <div id="timeline"></div>
  <div id="legend"></div>
```
new_string：
```
  <div id="timeline"></div>
  <div id="conflevel" style="margin-top:12px; display:flex; align-items:center;
       gap:12px; font-family:var(--mono); font-size:12px; color:var(--fg-dim);">
    <span>把握程度 ≥</span>
    <input id="confSlider" type="range" min="0" max="4" step="1" value="4"
           style="flex:1; max-width:320px;">
    <span id="confVal" style="color:var(--accent); font-weight:600; min-width:2.5em;">0.9</span>
    <span id="confSegCount"></span>
  </div>
  <div id="legend"></div>
```

## 修改 2：加 JS 常量与状态

`Edit` old_string：
```
let videos = [], curRel = null, doc = null, selected = -1, classNames = [];
```
new_string：
```
let videos = [], curRel = null, doc = null, selected = -1, classNames = [];
const CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9];
let confLevelIdx = 4; // 默认 0.9 档
```

## 修改 3：换档函数（插在 `function render()` 之前）

`Edit` old_string：
```
function render() {
  tlEl.innerHTML = "<div id='cursor' style='left:0'></div>";
```
new_string：
```
function levelKey() { return CONF_LEVELS[confLevelIdx].toFixed(1); }

function hasLevels() {
  return doc && doc.segment_levels &&
         Object.prototype.hasOwnProperty.call(doc.segment_levels, levelKey());
}

function applyLevel() {
  // 换档：把当前 segments 推进 undo 栈，再载入该档段（深拷贝，勿污染参考池）。
  if (!hasLevels()) return;
  pushUndo(snapNow());
  doc.segments = JSON.parse(JSON.stringify(doc.segment_levels[levelKey()]));
  selected = -1;
  render();
}

function syncConfUI() {
  const slider = $("confSlider");
  const enabled = doc && doc.segment_levels;
  slider.disabled = !enabled;
  $("conflevel").style.opacity = enabled ? "1" : "0.4";
  $("confVal").textContent = CONF_LEVELS[confLevelIdx].toFixed(1);
  $("confSegCount").textContent = doc ? `显示 ${doc.segments.length} 段` : "";
}

function render() {
  tlEl.innerHTML = "<div id='cursor' style='left:0'></div>";
```

## 修改 4：loadVideo 里初始化滑块 UI

`Edit` old_string：
```
  $("status").textContent = doc.approved ? "Approved ✅" : "";  // reflect the current video's real status
  render(); loadList();
}
```
new_string：
```
  $("status").textContent = doc.approved ? "Approved ✅" : "";  // reflect the current video's real status
  syncConfUI();
  render(); loadList();
}
```
说明：`doc.segments` 载入时已是 0.9 档（后端保证），滑块默认 idx=4（0.9）与之一致，无需在载入时强制换档。旧 doc 无 `segment_levels` → `syncConfUI` 置灰滑块。

## 修改 5：save 成功后刷新段数显示

`Edit` old_string：
```
  $("status").textContent = action === "approve" ? "Approved ✅"
    : "Saved (approval cleared, please re-Approve)";
  render(); loadList();
}
```
new_string：
```
  $("status").textContent = action === "approve" ? "Approved ✅"
    : "Saved (approval cleared, please re-Approve)";
  render(); loadList(); syncConfUI();
}
```

## 修改 6：绑定滑块事件

`Edit` old_string：
```
window.onresize = () => doc && render();
```
new_string：
```
window.onresize = () => doc && render();
$("confSlider").oninput = e => {
  confLevelIdx = parseInt(e.target.value, 10);
  applyLevel();
  syncConfUI();
};
```

## 验证（前端无自动化测试，手动确认逻辑正确性）

因为无法在此环境跑浏览器，做静态自检：
1. 用 `Read` 重新读改后的 `auto_presort/review_app.html`，确认 6 处修改都在、`<script>` 内 JS 语法配平（括号/引号/反引号成对）。
2. 确认这些被引用的符号都已存在于文件中：`pushUndo`、`snapNow`、`render`、`loadList`、`$`、`doc`、`api`。它们是文件里已有的函数/变量，你的新代码只是调用它们。
3. 确认没有引入未定义符号，没有重复声明 `CONF_LEVELS` / `confLevelIdx`。

不要跑 pytest（本任务不涉及 Python）。不要改其他文件。

## 硬约束
- 本仓库**不是 git 仓库**——不要执行 git 命令，不要 commit。
- 只改 `auto_presort/review_app.html`。
- 保持文件既有风格（2 空格缩进、中文注释、`$("id")` 取元素）。
- 中文字符（把握程度 / 显示 N 段 / 档）按原样写入，文件是 UTF-8。
