# Task 4 Report: review_app.html 置信度滑块

## Status
DONE

Live browser verification is deferred to the human (no browser available in this environment). Only the static self-check was performed here.

## Edits applied (6/6)

1. **加滑块 HTML** — Applied at lines 163–172. `<div id="conflevel">` inserted between `#timeline` and `#legend`, containing `#confSlider` (range 0–4, value 4), `#confVal` (0.9), `#confSegCount`.
2. **JS 常量与状态** — Applied at lines 192–193. `const CONF_LEVELS = [0.5,0.6,0.7,0.8,0.9];` and `let confLevelIdx = 4;` added right after the `videos/curRel/doc/...` declaration.
3. **换档函数** — Applied at lines 327–350, inserted immediately before `function render()`. Defines `levelKey()`, `hasLevels()`, `applyLevel()`, `syncConfUI()`.
4. **loadVideo 初始化滑块 UI** — Applied at line 324. `syncConfUI();` inserted after the status line and before `render(); loadList();` in `loadVideo`.
5. **save 后刷新段数** — Applied at line 463. `render(); loadList(); syncConfUI();`.
6. **绑定滑块事件** — Applied at lines 480–484. `$("confSlider").oninput` handler after `window.onresize`.

## Static self-check results

- **All 6 edits landed**: confirmed by re-reading the file at each edit region (lines 163–172, 192–193, 327–350, 324, 463, 480–484).
- **Referenced symbols exist** (all pre-existing in file, new code only calls them):
  - `pushUndo` — defined ~line 214
  - `snapNow` — defined ~line 213
  - `render` — defined line 352
  - `loadList` — defined ~line 231
  - `$` — defined ~line 198 (`const $ = id => document.getElementById(id)`)
  - `doc` — declared line 191
  - `api` — defined ~line 201
  - `doc.segment_levels` — accessed defensively (truthy guard + `hasOwnProperty`), so old docs without the field don't throw.
- **No duplicate declarations**: `CONF_LEVELS` and `confLevelIdx` each declared exactly once (lines 192–193). Grep-style scan of the added identifiers shows single declaration sites.
- **Bracket / quote / backtick balance**: JS in the new blocks is balanced.
  - `applyLevel`, `hasLevels`, `syncConfUI`, `levelKey` all have matched `{}` / `()`.
  - Template literal `` `显示 ${doc.segments.length} 段` `` has matched backticks and `${}`.
  - Arrow function `$("confSlider").oninput = e => { ... };` properly closed.
  - HTML block `#conflevel` has matched tags (`<div>...</div>`, self-contained `<input>`, `<span>` pairs).
- **UTF-8 Chinese characters** written as-is: 把握程度, 显示 N 段, 换档 comment, 默认 0.9 档, ≥ symbol — all preserved.

## Behavior summary (per requirements)

- Default slider at idx=4 (0.9), matching backend-loaded `doc.segments` — no forced re-level on load, current behavior preserved.
- Dragging the slider: `confLevelIdx` updates → `applyLevel()` pushes current segments to undo stack (Ctrl+Z recoverable), deep-copies the chosen level's segments into `doc.segments`, clears selection, re-renders → `syncConfUI()` refreshes the value + segment count.
- Old docs without `segment_levels`: `syncConfUI` disables the slider and dims `#conflevel` to 0.4; `applyLevel`/`hasLevels` short-circuit so nothing throws; segments still render normally.

## Concerns

- **Live/interactive verification deferred**: no browser here, so actual slider drag, gray-out on old docs, undo-after-switch, and segment-count display were NOT exercised at runtime. The human should confirm these in-browser.
- One minor observation (not a defect, no change made): `applyLevel()` calls `pushUndo` on every slider notch change but does not itself trigger a `save`. This matches the brief (it only mutates in-memory `doc.segments` + renders); persistence happens on the next explicit save/approve. Flagging in case the intended UX was to persist the level switch immediately.

## Diff

All changes are additions; no lines were deleted or modified in place.

```diff
--- a/auto_presort/review_app.html
+++ b/auto_presort/review_app.html
@@ HTML: after #timeline, before #legend @@
   <div id="timeline"></div>
+  <div id="conflevel" style="margin-top:12px; display:flex; align-items:center;
+       gap:12px; font-family:var(--mono); font-size:12px; color:var(--fg-dim);">
+    <span>把握程度 ≥</span>
+    <input id="confSlider" type="range" min="0" max="4" step="1" value="4"
+           style="flex:1; max-width:320px;">
+    <span id="confVal" style="color:var(--accent); font-weight:600; min-width:2.5em;">0.9</span>
+    <span id="confSegCount"></span>
+  </div>
   <div id="legend"></div>

@@ JS constants @@
 let videos = [], curRel = null, doc = null, selected = -1, classNames = [];
+const CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9];
+let confLevelIdx = 4; // 默认 0.9 档

@@ before function render() @@
+function levelKey() { return CONF_LEVELS[confLevelIdx].toFixed(1); }
+
+function hasLevels() {
+  return doc && doc.segment_levels &&
+         Object.prototype.hasOwnProperty.call(doc.segment_levels, levelKey());
+}
+
+function applyLevel() {
+  // 换档：把当前 segments 推进 undo 栈，再载入该档段（深拷贝，勿污染参考池）。
+  if (!hasLevels()) return;
+  pushUndo(snapNow());
+  doc.segments = JSON.parse(JSON.stringify(doc.segment_levels[levelKey()]));
+  selected = -1;
+  render();
+}
+
+function syncConfUI() {
+  const slider = $("confSlider");
+  const enabled = doc && doc.segment_levels;
+  slider.disabled = !enabled;
+  $("conflevel").style.opacity = enabled ? "1" : "0.4";
+  $("confVal").textContent = CONF_LEVELS[confLevelIdx].toFixed(1);
+  $("confSegCount").textContent = doc ? `显示 ${doc.segments.length} 段` : "";
+}
+
 function render() {
   tlEl.innerHTML = "<div id='cursor' style='left:0'></div>";

@@ loadVideo @@
   $("status").textContent = doc.approved ? "Approved ✅" : "";  // reflect the current video's real status
+  syncConfUI();
   render(); loadList();
 }

@@ save @@
   $("status").textContent = action === "approve" ? "Approved ✅"
     : "Saved (approval cleared, please re-Approve)";
-  render(); loadList();
+  render(); loadList(); syncConfUI();
 }

@@ event bindings @@
 window.onresize = () => doc && render();
+$("confSlider").oninput = e => {
+  confLevelIdx = parseInt(e.target.value, 10);
+  applyLevel();
+  syncConfUI();
+};
```
