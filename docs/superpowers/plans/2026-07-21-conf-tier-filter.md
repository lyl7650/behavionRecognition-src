# Confidence Tier Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add confidence tier filter to review UI sidebar for triaging videos by model confidence level.

**Architecture:** Single-feature enhancement - backend adds `conf_tier` field to `/api/videos`, frontend adds checkbox filters above video list with client-side filtering logic.

**Tech Stack:** Python stdlib http.server (backend), vanilla JS (frontend), pytest (testing)

## Global Constraints

- All UI text must be English
- Test command: `conda run --no-capture-output -n AR python -m pytest <path> -v`
- This repo is NOT a git repo - task completion = tests green, no commits needed
- No code changes without completing corresponding tests first (TDD)
- Default tier filter state: only 0.9-1.0 checked
- Preserve existing filter behavior (All/Labeled/Unapproved)

---

### Task 1: Backend - Add conf_tier Calculation

<thinking>
This task adds the conf_tier field calculation to the /api/videos endpoint. I'll write the test first, then implement the logic.

Files:
- Test: tests/test_autopresort_review_server.py (add test)
- Modify: auto_presort/review_server.py (add conf_tier calculation)

Interfaces:
- Consumes: store.list_docs (existing)
- Produces: /api/videos response with conf_tier field (string | null)

Steps:
1. Write failing test
2. Run test to verify it fails
3. Implement conf_tier calculation
4. Run test to verify it passes
5. Run full autopresort test suite
</thinking>

**Files:**
- Test: `tests/test_autopresort_review_server.py`
- Modify: `auto_presort/review_server.py:83-89` (the `/api/videos` endpoint)

**Interfaces:**
- Consumes: `store.list_docs(batch_dir)` returning list of doc dicts with `segment_levels` field
- Produces: `/api/videos` JSON response with added `conf_tier` field (string "0.9"|"0.8"|"0.7"|"0.6"|"0.5" or null)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_autopresort_review_server.py` at end of file:

```python
def test_api_videos_includes_conf_tier(tmp_path):
    """Verify /api/videos includes conf_tier field computed from segment_levels."""
    st = importlib.import_module("auto_presort.store")
    
    # Video 1: has segments at 0.8 level (not at 0.9) → tier = "0.8"
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v1.mp4",
        "source_id": "s1", "duration": 10.0, "approved": False,
        "segments": [], "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":3,"label":"eat","confidence":0.6,"source":"model"}],
            "0.6": [{"start":1,"end":3,"label":"eat","confidence":0.65,"source":"model"}],
            "0.7": [{"start":2,"end":4,"label":"eat","confidence":0.75,"source":"model"}],
            "0.8": [{"start":2,"end":4,"label":"eat","confidence":0.85,"source":"model"}],
            "0.9": []  # empty at 0.9 → highest non-empty is 0.8
        },
        "meta": {"class_names": ["eat"]},
    })
    
    # Video 2: has segments at 0.9 level → tier = "0.9"
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v2.mp4",
        "source_id": "s2", "duration": 12.0, "approved": True,
        "segments": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
        "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
            "0.9": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
        },
        "meta": {"class_names": ["drink"]},
    })
    
    # Video 3: no segment_levels (old format) → tier = null
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-1", "rel_path": "v3.mp4",
        "source_id": "s3", "duration": 8.0, "approved": False,
        "segments": [], "gaps": [], "meta": {"class_names": ["eat"]},
    })
    
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            items = json.loads(r.read())
        
        assert len(items) == 3
        
        v1 = next(x for x in items if x["rel_path"] == "v1.mp4")
        assert v1["conf_tier"] == "0.8", "Should pick highest non-empty tier"
        
        v2 = next(x for x in items if x["rel_path"] == "v2.mp4")
        assert v2["conf_tier"] == "0.9", "Should pick 0.9 when available"
        
        v3 = next(x for x in items if x["rel_path"] == "v3.mp4")
        assert v3["conf_tier"] is None, "No segment_levels → null tier"
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n AR python -m pytest tests/test_autopresort_review_server.py::test_api_videos_includes_conf_tier -v`

Expected: FAIL with "KeyError: 'conf_tier'" or similar (field doesn't exist yet)

- [ ] **Step 3: Implement conf_tier calculation**

In `auto_presort/review_server.py`, replace lines 83-89 (the `/api/videos` handler body):

```python
            elif parsed.path == "/api/videos":
                docs = store.list_docs(batch_dir)
                items = []
                for d in docs:
                    # Calculate conf_tier: highest level with non-empty segments
                    conf_tier = None
                    if "segment_levels" in d and d["segment_levels"]:
                        for level in [0.9, 0.8, 0.7, 0.6, 0.5]:
                            key = f"{level:.1f}"
                            if key in d["segment_levels"] and len(d["segment_levels"][key]) > 0:
                                conf_tier = key
                                break
                    items.append({
                        "rel_path": d["rel_path"],
                        "approved": d["approved"],
                        "duration": d["duration"],
                        "n_segments": len(d["segments"]),
                        "class_names": d.get("meta", {}).get("class_names", []),
                        "conf_tier": conf_tier,
                    })
                self._json(items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n AR python -m pytest tests/test_autopresort_review_server.py::test_api_videos_includes_conf_tier -v`

Expected: PASS

- [ ] **Step 5: Run full autopresort test suite**

Run: `conda run --no-capture-output -n AR python -m pytest tests/ -v -k "autopresort"`

Expected: All tests PASS (new test + all existing tests)

---

### Task 2: Frontend - Add Tier Filter UI

<thinking>
This task adds the HTML markup and JavaScript logic for the tier filter. No automated test (no headless browser), so it's implement → manual verification.

Files:
- Modify: auto_presort/review_app.html (add HTML between vfilter and vlist, add JS logic)

Interfaces:
- Consumes: videos array from /api/videos with conf_tier field (added in Task 1)
- Produces: filtered video list display based on checked tiers

Steps:
1. Add HTML markup for tier checkboxes
2. Add JS: tier count calculation
3. Add JS: tier filtering logic in loadList
4. Add JS: checkbox event bindings
5. Manual browser verification steps
</thinking>

**Files:**
- Modify: `auto_presort/review_app.html`

**Interfaces:**
- Consumes: `videos` array from Task 1's `/api/videos` endpoint, each item having `conf_tier` field
- Produces: Filtered video list in `#vlist` based on checked tier checkboxes

- [ ] **Step 1: Add HTML markup for tier checkboxes**

In `auto_presort/review_app.html`, find the line with `<div id="vlist"></div>` (currently line 158).

Insert BEFORE `<div id="vlist"></div>`:

```html
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
```

- [ ] **Step 2: Add tier count calculation to loadList**

In `auto_presort/review_app.html`, find the `async function loadList()` definition (currently line 233).

At the START of the function body (right after `videos = await api("/api/videos");`), add:

```javascript
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
```

- [ ] **Step 3: Add tier filtering logic to loadList**

In the same `loadList()` function, find the lines that compute `shown` (currently around line 241):

```javascript
  const shown = videos.filter(pred);
```

Replace with:

```javascript
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

- [ ] **Step 4: Bind checkbox change events**

At the END of the `<script>` block in `auto_presort/review_app.html` (currently before line 493, after the existing event bindings), add:

```javascript
for (const cb of document.querySelectorAll(".tierCheck")) {
  cb.onchange = () => loadList();
}
```

- [ ] **Step 5: Manual browser verification**

Start the review server and verify in browser:

```
cd E:\ActionRecognition\behavionRecognition-src
conda run --no-capture-output -n AR python -m auto_presort.review_server <batch_dir>
```

Open browser to `http://127.0.0.1:8765`

**Verification checklist:**
1. Tier filter appears below All/Labeled/Unapproved buttons
2. Only "0.9-1.0" checkbox is checked by default
3. Tier counts display correctly (sum equals total videos)
4. Video list shows only 0.9-1.0 tier videos on page load
5. Check "0.8-0.9" → list includes both 0.9 and 0.8 videos
6. Uncheck all → list shows "No videos under this filter"
7. Check "No model" → shows old-format videos (if any exist)
8. Interaction with All/Labeled/Unapproved: select "Labeled" + only "0.9-1.0" → list shows intersection
9. All UI text is English (no Chinese characters)

Expected: All 9 verification points pass

---

## Self-Review Checklist

- [ ] No placeholders (TBD/TODO/etc.) - all code complete
- [ ] File paths are exact and absolute where needed
- [ ] Commands include expected output
- [ ] Test code matches implementation signatures
- [ ] Global constraints respected (English text, conda test command, no git)
- [ ] Each task is independently testable
