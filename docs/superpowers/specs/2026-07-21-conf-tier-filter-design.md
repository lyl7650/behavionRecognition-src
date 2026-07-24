# Confidence Tier Filter for auto_presort Review

## Goal

Add a confidence tier filter to the review UI sidebar so users can triage videos by model confidence: quickly approve high-confidence batches (0.9-1.0) first, then focus on lower-confidence videos that need careful review.

## Architecture

Single-feature enhancement to the existing auto_presort review system. Two files modified:

1. **Backend** (`auto_presort/review_server.py`): Add `conf_tier` field to `/api/videos` response
2. **Frontend** (`auto_presort/review_app.html`): Add confidence tier checkboxes below existing filters, implement client-side filtering

No changes to 2a (labeling) or 2c (clip generation). Data source is existing `segment_levels` from 2a output.

## Confidence Tier Calculation (Backend)

### Definition

Each video is assigned a single `conf_tier` string representing the highest confidence level at which the model produced segments:

- `"0.9"` → video has segments in `segment_levels["0.9"]`
- `"0.8"` → highest tier with segments is `segment_levels["0.8"]` (no segments at 0.9)
- `"0.7"`, `"0.6"`, `"0.5"` → similarly
- `null` → video has no `segment_levels` field (schema v1 / no model output)

### Calculation Logic

In `review_server.py`, modify the `/api/videos` endpoint (lines 83-89):

```python
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

**Key properties:**
- Tier assignment is deterministic and stable (only depends on `segment_levels`, never on `doc.segments` which can be human-edited)
- Each video falls into exactly one tier (or null)
- Calculation happens on every `/api/videos` call (cheap: one loop per doc, no video I/O)

## UI Design (Frontend)

### Placement

Insert confidence tier filter between existing filter buttons and video list in the sidebar:

```
#sidebar structure:
  <h3>Videos</h3>
  <div id="vfilter">  <!-- existing All/Labeled/Unapproved buttons -->
  </div>
  <div id="confTierFilter">  <!-- NEW: confidence tier checkboxes -->
  </div>
  <div id="vlist"></div>  <!-- video list -->
```

### Visual Design

Match existing sidebar style:

- Label: "Confidence tier:" (12px uppercase tracking, same as sidebar h3 style but smaller)
- 6 checkboxes in vertical list:
  - `0.9-1.0 (N)` ← checked by default
  - `0.8-0.9 (N)`
  - `0.7-0.8 (N)`
  - `0.6-0.7 (N)`
  - `0.5-0.6 (N)`
  - `No model (N)` ← for null tier
- Count `(N)` auto-updates from `videos` array
- Styling: 12px mono font, color `var(--fg-dim)`, checkbox uses browser default (keep it simple)

### Markup

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

## Filter Behavior

### Interaction with Existing Filters

Confidence tier filter **intersects** with existing All/Labeled/Unapproved filter:

1. Apply existing filter (All/Labeled/Unapproved) to get candidate set
2. Apply tier filter: keep only videos whose `conf_tier` matches any checked tier checkbox
3. Render resulting list

Example: if "Labeled" is active and only "0.9" tier is checked → show videos that are both labeled AND tier 0.9.

### JavaScript Logic

Modify `loadList()` function (currently line 233-261):

```javascript
async function loadList() {
  videos = await api("/api/videos");
  
  // Update tier counts
  const tierCounts = {"0.9":0, "0.8":0, "0.7":0, "0.6":0, "0.5":0, "null":0};
  for (const v of videos) {
    tierCounts[v.conf_tier || "null"]++;
  }
  for (const tier of ["0.9", "0.8", "0.7", "0.6", "0.5", "null"]) {
    const span = document.querySelector(`.tierCount[data-tier="${tier}"]`);
    if (span) span.textContent = `(${tierCounts[tier]})`;
  }
  
  // Update main filter counts (existing logic)
  $("fcount-all").textContent = videos.length;
  $("fcount-labeled").textContent = videos.filter(v => v.n_segments > 0).length;
  $("fcount-unapproved").textContent = videos.filter(v => !v.approved).length;
  
  // Apply filters
  const pred = filter === "labeled" ? (v => v.n_segments > 0)
             : filter === "unapproved" ? (v => !v.approved)
             : (() => true);
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
  
  // Render list (existing logic)
  const el = $("vlist"); el.innerHTML = "";
  if (!shown.length) {
    el.innerHTML = "<div style='padding:12px;color:var(--fg-dim);font-size:12px'>No videos under this filter</div>";
    loadStats(); return;
  }
  for (const v of shown) {
    // ... existing render code
  }
  loadStats();
}
```

Bind checkbox events (add at end of script):

```javascript
for (const cb of document.querySelectorAll(".tierCheck")) {
  cb.onchange = () => loadList();
}
```

### Edge Cases

- **All tiers unchecked**: list shows empty with "No videos under this filter" message (existing behavior)
- **null tier**: videos with no `segment_levels` (schema v1 or no model output) are grouped under "No model", unchecked by default
- **Empty tier**: if a tier has 0 videos, count shows `(0)`, checkbox still functional (user can check it, just shows nothing)

## Testing Strategy

### Manual Browser Tests

1. **Default state**: Open review UI → only 0.9-1.0 tier checked → list shows only high-confidence videos
2. **Check multiple tiers**: Check 0.8 and 0.7 → list includes videos from both tiers
3. **Interaction with All/Labeled/Unapproved**: Select "Labeled" + check only 0.5 → list shows labeled videos that are tier 0.5
4. **Uncheck all tiers**: List becomes empty with "No videos" message
5. **null tier videos**: Verify old-format docs appear under "No model" (unchecked by default)
6. **Counts**: Verify tier counts sum to total videos (accounting for each video appearing in exactly one tier)

### Backend Unit Test

Add to `tests/test_autopresort_review_server.py`:

```python
def test_api_videos_includes_conf_tier(tmp_path):
    # Seed: one video with segment_levels (tier 0.8), one without (tier null)
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v1.mp4",
        "source_id": "s1", "duration": 10.0, "approved": False,
        "segments": [], "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":3,"label":"eat","confidence":0.6,"source":"model"}],
            "0.8": [{"start":2,"end":4,"label":"eat","confidence":0.85,"source":"model"}],
            "0.9": []  # empty at 0.9 → tier should be 0.8
        },
        "meta": {"class_names": ["eat"]},
    })
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-1", "rel_path": "v2.mp4",
        "source_id": "s2", "duration": 8.0, "approved": False,
        "segments": [], "gaps": [], "meta": {"class_names": ["eat"]},
    })
    
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            items = json.loads(r.read())
        assert len(items) == 2
        v1 = next(x for x in items if x["rel_path"] == "v1.mp4")
        v2 = next(x for x in items if x["rel_path"] == "v2.mp4")
        assert v1["conf_tier"] == "0.8"  # highest non-empty tier
        assert v2["conf_tier"] is None    # no segment_levels
    finally:
        server.shutdown()
```

No pytest for frontend (no headless browser setup). Frontend verified manually per list above.

## Global Constraints

- **English only**: All UI text in English (tier labels, "No model", existing "No videos" message already English)
- **No breaking changes**: Existing filter buttons (All/Labeled/Unapproved) behavior unchanged, tier filter adds on top
- **Backward compatibility**: Old-format docs (schema v1, no `segment_levels`) handled gracefully via null tier
- **User approval required**: No code changes without explicit user consent (per CLAUDE.md)
- **Conda test command**: `conda run --no-capture-output -n AR python -m pytest` (per memory)

## Files Modified

1. `auto_presort/review_server.py` (~15 lines added to `/api/videos` handler)
2. `auto_presort/review_app.html` (~60 lines: HTML markup + JS filter logic)
3. `tests/test_autopresort_review_server.py` (~30 lines: one new test)

Total scope: ~105 lines, 3 files, single feature, no external dependencies.
