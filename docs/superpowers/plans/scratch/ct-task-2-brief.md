# Task 2 Brief: Frontend — Confidence Tier Filter UI

这是你的需求来源，使用其中的确切代码（逐字照抄，勿改）。只改一个文件：`auto_presort/review_app.html`。

## 背景
审核页左侧边栏现有三个过滤按钮（All / Labeled / Unapproved）。后端 `/api/videos` 现在给每个视频返回了一个新字段 `conf_tier`（字符串 `"0.9"`…`"0.5"`，或 null）。要在过滤按钮**下面、视频列表上面**加一组"置信度档位"复选框，按 `conf_tier` 筛选视频列表。默认只勾 0.9-1.0 档。null 档归入 "No model"。跟上方 All/Labeled/Unapproved 过滤叠加（取交集）。

用 `Edit` 工具做下面 4 处修改，每处 old_string 都能在文件中唯一匹配。改前先 `Read` 整个文件确认锚点存在。

## 修改 1：加复选框 HTML

`Edit` old_string：
```
  <div id="vlist"></div>
```
new_string：
```
  <div id="confTierFilter" style="margin:0 4px 12px; padding-top:8px; border-top:1px solid var(--line);">
    <div style="font-size:11px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--fg-dim); margin-bottom:8px;">
      Confidence tier:
    </div>
    <label style="display:block; margin-bottom:4px; font-family:var(--mono); font-size:12px; color:var(--fg-dim); cursor:pointer;">
      <input type="checkbox" class="tierCheck" data-tier="0.9" checked> 0.9-1.0 <span class="tierCount" data-tier="0.9"></span>
    </label>
    <label style="display:block; margin-bottom:4px; font-family:var(--mono); font-size:12px; color:var(--fg-dim); cursor:pointer;">
      <input type="checkbox" class="tierCheck" data-tier="0.8"> 0.8-0.9 <span class="tierCount" data-tier="0.8"></span>
    </label>
    <label style="display:block; margin-bottom:4px; font-family:var(--mono); font-size:12px; color:var(--fg-dim); cursor:pointer;">
      <input type="checkbox" class="tierCheck" data-tier="0.7"> 0.7-0.8 <span class="tierCount" data-tier="0.7"></span>
    </label>
    <label style="display:block; margin-bottom:4px; font-family:var(--mono); font-size:12px; color:var(--fg-dim); cursor:pointer;">
      <input type="checkbox" class="tierCheck" data-tier="0.6"> 0.6-0.7 <span class="tierCount" data-tier="0.6"></span>
    </label>
    <label style="display:block; margin-bottom:4px; font-family:var(--mono); font-size:12px; color:var(--fg-dim); cursor:pointer;">
      <input type="checkbox" class="tierCheck" data-tier="0.5"> 0.5-0.6 <span class="tierCount" data-tier="0.5"></span>
    </label>
    <label style="display:block; margin-bottom:4px; font-family:var(--mono); font-size:12px; color:var(--fg-dim); cursor:pointer;">
      <input type="checkbox" class="tierCheck" data-tier="null"> No model <span class="tierCount" data-tier="null"></span>
    </label>
  </div>
  <div id="vlist"></div>
```

## 修改 2：loadList 里加档位计数

在 `async function loadList()` 里，找到这一行（`videos = await api("/api/videos");` 后面紧跟的一行是 `$("fcount-all")...`）：

`Edit` old_string：
```
  videos = await api("/api/videos");
  $("fcount-all").textContent = videos.length;
```
new_string：
```
  videos = await api("/api/videos");
  // Update tier counts
  const tierCounts = {"0.9":0, "0.8":0, "0.7":0, "0.6":0, "0.5":0, "null":0};
  for (const v of videos) {
    const tier = v.conf_tier || "null";
    tierCounts[tier]++;
  }
  for (const tier of ["0.9", "0.8", "0.7", "0.6", "0.5", "null"]) {
    const span = document.querySelector(`.tierCount[data-tier="${tier}"]`);
    if (span) span.textContent = `(${tierCounts[tier]})`;
  }
  $("fcount-all").textContent = videos.length;
```

## 修改 3：loadList 里加档位过滤

`Edit` old_string：
```
  const shown = videos.filter(pred);
```
new_string：
```
  let shown = videos.filter(pred);

  // Apply tier filter
  const checkedTiers = Array.from(document.querySelectorAll(".tierCheck:checked"))
    .map(cb => cb.dataset.tier);
  if (checkedTiers.length > 0) {
    shown = shown.filter(v => {
      const tier = v.conf_tier || "null";
      return checkedTiers.includes(tier);
    });
  }
```

## 修改 4：绑定复选框事件

在 `<script>` 块末尾，找到已有的这段过滤按钮绑定（在 `loadList().then(...)` 之前）：

`Edit` old_string：
```
for (const btn of document.querySelectorAll(".fbtn")) {
  btn.onclick = () => {
    filter = btn.dataset.f;
    for (const b of document.querySelectorAll(".fbtn")) b.classList.toggle("active", b === btn);
    loadList();
  };
}
```
new_string：
```
for (const btn of document.querySelectorAll(".fbtn")) {
  btn.onclick = () => {
    filter = btn.dataset.f;
    for (const b of document.querySelectorAll(".fbtn")) b.classList.toggle("active", b === btn);
    loadList();
  };
}
for (const cb of document.querySelectorAll(".tierCheck")) {
  cb.onchange = () => loadList();
}
```

## 验证（前端无自动化测试，做静态自检）

无法在此环境跑浏览器，做静态自检：
1. 用 `Read` 重新读改后的 `auto_presort/review_app.html`，确认 4 处修改都在。
2. 确认 `<script>` 内 JS 语法配平：修改 3 把原来的 `const shown` 改成了 `let shown`（因为后面要重新赋值），确认文件里没有别处再 `const shown` 重复声明、也没有对 `shown` 的其它冲突赋值。
3. 确认 `loadList` 函数体后续用的还是 `shown`（渲染列表那段 `for (const v of shown)`），变量名一致。
4. 确认没有引入未定义符号；`videos`、`pred`、`$`、`loadList` 都是文件里已有的。
5. 确认新加的所有界面文字都是英文（Confidence tier / No model / 0.9-1.0 等），无中文。

不要跑 pytest（本任务不涉及 Python）。不要改其他文件。

## 硬约束
- 本仓库**不是 git 仓库**——不要执行 git 命令，不要 commit。
- 只改 `auto_presort/review_app.html`。
- 保持文件既有风格（2 空格缩进、`$("id")` 取元素、`var(--xxx)` CSS 变量）。
- 界面文字全英文，无中文、无 emoji。
