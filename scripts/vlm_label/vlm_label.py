r"""Qwen3-VL labeling for rabbit clips -> 9-class behaviour timeline JSON.

Per window: sample N frames evenly (W=6s / stride=3s default), resize so the
long side <= 448 px, one chat completion per window with a 9-way prompt
(Eating/Drinking/Grooming/Digging/BellyUp/Active/Resting/None/Unknown).

Answer format (rabbit-9class-json-v1): the label word first, then one valid
single-line JSON object --
  Grooming {"label": "Grooming", "confidence": 0.86, "frame_labels": [...],
            "observed_behaviours": [...], "visible_frame_count": 8,
            "supporting_frame_indices": [...], "mixed_behaviours": true,
            "evidence": "...", "review_reason": "none"}
so window confidence still comes from the real first-token softmax over the
9 candidate words; the JSON fields are stored per window as a cross-check
(self_confidence) plus the frame-level detail (frame_labels, evidence,
review_reason, ...). The frame count in the prompt is parameterized: the
majority-support rule becomes "at least N//2+1 frames" (8 frames -> 5).

Docs mirror the .xclip-*/ schema (segments/gaps share autopresort field
names) so scripts.vlm_label.vlm_review can serve two model dirs side by side.
No comparison against the dashboard/repvit labels is computed or stored.

Resume-safe: finished clips are skipped; each clip doc is written atomically.
Deadline-safe: env VLM_DEADLINE_EPOCH stops the run before that epoch second
(manifest.incomplete=true), so the morning always has a result.

Usage (venv python, cwd = project root):
  python -m scripts.vlm_label.vlm_label --dir G:\...\rabbit \
      --tag qwen3vl-8b-v3 --model-id Qwen/Qwen3-VL-8B-Instruct --dtype bf16
  python -m scripts.vlm_label.vlm_label --dir G:\...\rabbit \
      --tag qwen3vl-32b-v3 --model-id Qwen/Qwen3-VL-32B-Instruct \
      --dtype bnb4 --gpu-mem-gib 17
"""
import argparse
import hashlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")
LABELS = ["eating", "drinking", "grooming", "digging", "bellyup",
          "active", "resting", "none", "unknown"]
WORDS = ["Eating", "Drinking", "Grooming", "Digging", "BellyUp",
         "Active", "Resting", "None", "Unknown"]
SCHEMA_VERSION = "vlm-qwen3vl-3"
PROMPT_VERSION = "rabbit-9class-json-v1"
# compact variant (rabbit-9class-compact-v1): same 9 classes, one-line answer
# "<Label> {\"confidence\": x}" only -- no frame-level detail JSON. Schema v4.
SCHEMA_VERSION_COMPACT = "vlm-qwen3vl-4"
PROMPT_VERSION_COMPACT = "rabbit-9class-compact-v1"
# per-window JSON output is ~150-250 tokens; 512 leaves head-room for the
# one retry without runaway generation on a stuck model
DEFAULT_MAX_NEW_TOKENS = 512

# rabbit-9class-json-v1: full 9-class spec. Placeholders {NFRAMES},
# {MAJORITY}, {DURATION}, {TIMESTAMPS}, {SAMPLING}, {TARGET}, {SCENE} are
# filled per window by build_window_prompt() (plain .replace, no .format:
# the example JSON below contains literal braces).
TEXT_PROMPT = """TASK

You are assisting with video annotation for a pet rabbit behaviour dataset.

Produce a pre-label proposal, not a verified ground-truth label.

Classify the target rabbit's dominant observed behaviour across the supplied frames using exactly one of these labels:

Eating, Drinking, Grooming, Digging, BellyUp, Active, Resting, None, Unknown.

INPUT

- The supplied images are chronologically ordered samples from one short video clip.
- Number of supplied frames: {NFRAMES}.
- Clip duration: {DURATION} seconds.
- Frame timestamps (seconds from clip start): {TIMESTAMPS}.
- Sampling method: {SAMPLING}.
- Target rabbit: {TARGET}.
- Known scene information: {SCENE}.

Classify only the target rabbit.

Use the supplied visual observations and explicitly provided scene information only.

Do not invent actions occurring between sampled frames.
Do not infer intentions, emotional states, sleep, or medical conditions.

If multiple rabbits are present and the target cannot be reliably distinguished, return Unknown.

LABEL DEFINITIONS

1. Eating

The target visibly ingests identifiable food, including hay, grass, pellets, or other identifiable food.

Require clear feeding evidence, such as:
- The mouth visibly taking or biting food.
- Food visibly entering the mouth.
- Repeated chewing in a clearly supported food-consumption context.

A lowered head, proximity to food, sniffing food, or mouth movement without a clear feeding context is not sufficient.

Do not confuse eating with licking the body, washing the face, grooming, or chewing non-food objects.

If food consumption cannot be established from the supplied observations, do not infer Eating.
Use another supported label or Unknown.

2. Drinking

The target visibly drinks from an identifiable water source, such as a water bowl or a water bottle.

Require clear drinking evidence, such as:
- Visible lapping at the water surface.
- Mouth contact with a water-bottle nozzle together with repeated drinking-related mouth action.

Being near a water source, looking at it, sniffing it, or lowering the head over a bowl is not sufficient.

Do not assume that a container holds water based only on its shape.
Use supplied scene information when available, but proximity to a known water source alone is still not sufficient.

If the mouth-water interaction is obscured or the drinking action cannot be established, do not infer Drinking.
Use another supported label or Unknown.

3. Grooming

The target visibly performs self-grooming, including:
- Licking its own fur or body.
- Washing its face with its paws.
- Scratching its own body.

Require clear self-directed cleaning or scratching action.

A lowered head, mouth near the body, or an unclear chewing-like movement alone is not sufficient.

Do not confuse grooming with eating or drinking.

This definition covers self-grooming only. Grooming another animal is outside this label definition.

Grooming is a separate class and must not be labelled Active or Resting, even when the target remains in one location or lies down while grooming.

4. Digging

The target repeatedly paws, scrapes, or digs at the ground, bedding, litter, or another surface with its front paws.

Require a repeated digging or scraping action supported by the supplied observations.

Visible movement of the surface material may support the label, but is not required.

Walking, a single paw placement, stretching, or merely standing on bedding is not sufficient.

Do not infer elimination or toileting from digging alone.

Digging is a separate class and must not be labelled Active, even when the target does not change location.

5. BellyUp

The target maintains a clearly observable back-down, belly-up posture, with the abdomen facing upward.

Require clear evidence of this posture. Visible abdomen alone is not sufficient if the target is merely lying on its side.

Ordinary side-lying, stretched-out lying, or a brief transition through a belly-up position is not sufficient.

This is a posture label, not an inference about comfort, sleep, playfulness, or health.

When this posture is clear and no other specific behaviour is being performed, use BellyUp instead of Resting or Active.

If the target is clearly grooming, eating, or drinking while belly-up, use the specific behaviour label instead.

6. Active

The target shows clear locomotion or sustained active behaviour, including:
- Hopping, walking, or running.
- Rearing up.
- Repeated exploratory head or body movements.
- Rolling without maintaining a belly-up posture.
- Other clearly observable activity not covered by a more specific supported label.

The target does not need to change location to be Active.

Standing or sitting posture alone is not sufficient. Normal breathing or an isolated small ear movement is not sufficient.

Use Eating, Drinking, Grooming, Digging, or BellyUp instead when that specific label is clearly supported.

Do not use Active as a fallback for ambiguous, obscured, or unrecognised behaviour.

7. Resting

The target maintains a resting posture, such as:
- Sitting quietly.
- A loaf posture.
- Side-lying.
- Lying with limbs tucked.
- Lying stretched out.

There must be no clear locomotion or sustained active behaviour in the supporting observations.

Normal breathing and occasional small movements are allowed.
Do not require absolute immobility.
Do not infer sleep.

Do not choose Resting during eating, drinking, grooming, digging, or sustained exploratory movements.

Use BellyUp instead when a clear back-down, belly-up posture is maintained without another specific behaviour.

Do not treat missing or obscured observations as evidence of Resting.

8. None

The target rabbit is not visible in any of the supplied frames.

If the target rabbit is visible in at least one frame, never choose None.

None describes the absence of the target from the supplied observations. It does not establish that the target was absent during unsampled moments between frames.

Do not use None for a visible rabbit whose behaviour cannot be determined.

If another rabbit is visible but the target cannot be reliably identified or distinguished, use Unknown rather than None.

9. Unknown

The target rabbit is visible, or a visible rabbit may be the target, but a reliable dominant behaviour label cannot be assigned.

Use Unknown when:
- The target is too small, blurred, partially visible, or occluded to establish its behaviour.
- The target cannot be reliably distinguished from another rabbit.
- Eating, drinking, grooming, or digging interactions are ambiguous.
- Too few observations support a reliable dominant behaviour.
- Multiple behaviours appear without a clear dominant behaviour.
- The observed behaviour is outside the supported label definitions.
- Sampling or missing observations prevent a reliable decision.

Do not guess a specific behaviour merely to avoid Unknown.

Unknown is an annotation abstention. It is not equivalent to the absence of the target.

DECISION RULES

1. Check target visibility and identity first.

   If the target is not visible in any supplied frame and its absence can be established, return None.

   If a visible rabbit may be the target but its identity cannot be resolved, return Unknown.

2. Assign one provisional label to each supplied frame.

   Use neighbouring frames as temporal context, but do not invent unobserved actions.

   Use NotVisible when the target is absent from an individual frame.
   Use Unknown when the target is visible but its behaviour cannot be established.

   If target identity is uncertain in a frame, use Unknown rather than NotVisible.

3. Use specific behaviour labels instead of general activity or posture labels when the specific action is clearly supported.

   Examples:
   - Eating while sitting still -> Eating.
   - Drinking while otherwise stationary -> Drinking.
   - Grooming while lying down or belly-up -> Grooming.
   - Digging without changing location -> Digging.
   - Maintaining a clear belly-up posture without another specific behaviour -> BellyUp.
   - Rolling without maintaining a belly-up posture -> Active.

   If two specific behaviours cannot be distinguished, use Unknown rather than applying an arbitrary priority.

4. Do not classify the target as Active solely because it is standing or sitting upright.

   Do not classify normal breathing or an isolated small ear movement as Active.

5. For exactly {NFRAMES} uniformly sampled frames, propose a behaviour label only when at least {MAJORITY} frames support the same behaviour.

   Count support across all {NFRAMES} supplied frames, not only the visible or classifiable frames.

   Otherwise, return Unknown and request review.

6. Do not use NotVisible or Unknown observations as evidence for Resting or any other behaviour.

7. Do not apply a clip-level tie-break preference for any label.

   If no behaviour meets the dominant-behaviour requirement, return Unknown even if one behaviour is more specific than another.

8. If the timestamps are not uniformly spaced, do not use raw frame counts as a proxy for duration.

   For a clip containing a visible target, return Unknown with review_reason set to nonuniform_sampling.

   This sampling restriction does not prevent a None label when the target is demonstrably absent from all supplied frames.

9. Record all clearly supported behaviours in observed_behaviours, even when they are not the dominant label.

   Do not include None, Unknown, or NotVisible in this list.

10. Set mixed_behaviours to true when more than one supported behaviour appears in the supplied observations.

    A mixed clip may still receive a dominant behaviour label if that behaviour meets the support requirement.

11. Confidence is a self-reported estimate of support for the output decision from the supplied observations.

    It is not a calibrated probability of correctness. A high confidence value must not override insufficient evidence or the dominant-behaviour requirement.

12. Evidence must describe visible actions, interactions, or posture and identify relevant frame indices when possible.

    Do not describe inferred intentions or unobserved events.

OUTPUT FORMAT

Answer in EXACTLY this format on one line and nothing else:

<Label> {json object}

- First write exactly one allowed clip label word (no punctuation before it), then one space, then exactly one valid JSON object.
- The leading label word and the JSON "label" field must be identical.
- Do not include markdown, code fences, or any additional text.
- Keep the JSON on a single line and keep "evidence" under 30 words.

Required fields:
- label
- confidence
- frame_labels
- observed_behaviours
- visible_frame_count
- supporting_frame_indices
- mixed_behaviours
- evidence
- review_reason

Allowed clip labels:
Eating, Drinking, Grooming, Digging, BellyUp, Active, Resting, None, Unknown.

Allowed frame labels:
Eating, Drinking, Grooming, Digging, BellyUp, Active, Resting, Unknown, NotVisible.

Allowed review_reason values:
none, poor_visibility, ambiguous_interaction, mixed_behaviours, insufficient_support, identity_uncertain, unsupported_behaviour, nonuniform_sampling.

Validation rules:
- label must contain exactly one allowed clip label.
- confidence must be a number between 0.0 and 1.0.
- frame_labels must contain exactly {NFRAMES} values in chronological order.
- observed_behaviours must contain unique supported behaviour labels in their order of first appearance.
- visible_frame_count must count only frames in which the target is confidently identified as visible.
- visible_frame_count must be an integer from 0 to {NFRAMES}.
- supporting_frame_indices must use 1-based frame indices.
- For a behaviour label, supporting_frame_indices must identify all frames supporting that label.
- For None or Unknown, supporting_frame_indices must be empty.
- For None, all frame_labels must be NotVisible, visible_frame_count must be 0, observed_behaviours must be empty, mixed_behaviours must be false, and review_reason must be none.
- For Unknown, review_reason must identify the main reason a reliable dominant behaviour could not be assigned.
- For an accepted dominant behaviour label, review_reason must be none.
- mixed_behaviours must be true if observed_behaviours contains more than one behaviour, and false otherwise.
- evidence must be a concise description of visible evidence or the reason for abstaining.

Example output:

Grooming {"label": "Grooming", "confidence": 0.86, "frame_labels": ["Resting", "Grooming", "Grooming", "Grooming", "Grooming", "Grooming", "Grooming", "Resting"], "observed_behaviours": ["Resting", "Grooming"], "visible_frame_count": 8, "supporting_frame_indices": [2, 3, 4, 5, 6, 7], "mixed_behaviours": true, "evidence": "In frames 2-7, the target visibly licks its own fur, supporting self-grooming.", "review_reason": "none"}"""

RETRY_SUFFIX = (
    " You must output exactly one line: first one allowed clip label word "
    "(Eating, Drinking, Grooming, Digging, BellyUp, Active, Resting, None, "
    "Unknown), then one space, then one valid single-line JSON object with "
    "all nine required fields, and nothing else."
)

DEFAULT_TARGET = "the single pet rabbit in the clip"
DEFAULT_SCENE = "none"

# rabbit-9class-compact-v1: user-supplied condensed 9-class spec. Same label
# set as the full spec but the answer carries only the label + self-reported
# confidence (no frame_labels / evidence / review_reason), which makes each
# window ~10x cheaper to generate. --prompt compact selects this; the leading
# label word still feeds the first-token softmax exactly like the full spec.
PROMPT_COMPACT = """These 8 frames are time-ordered samples from one short clip
of a pet rabbit. Classify the rabbit's dominant observed state
with exactly one label.
Use these definitions:
- None: no rabbit is visible in any frame. If a rabbit is visible
in at least one frame, never choose None.
- Eating: visibly taking or chewing identifiable food, including
hay, grass or pellets. Head-down posture or proximity to food
alone is not enough. Do not confuse eating with grooming.
- Drinking: actively drinking at a water bowl or bottle.
Being near a water source is not enough.
- Grooming: visibly licking its own fur/body, washing its face
with its paws, or scratching its own body.
- Digging: repeatedly scraping or digging at a surface with
the front paws. Walking or a single paw placement is not enough.
- BellyUp: maintaining a clear back-down, abdomen-up posture.
Side-lying or briefly rolling through this position is not enough.
If grooming while belly-up, choose Grooming.
- Active: clear movement or sustained activity, including hopping,
running, rearing, exploring or rolling. Do not use Active for
Eating, Drinking, Grooming or Digging.
- Resting: sitting quietly, loafing or lying down without sustained
activity. Breathing and small occasional movements are allowed.
Use BellyUp for a clear maintained belly-up posture.
- Unknown: a rabbit is visible, but poor visibility, ambiguous
behaviour or a tie prevents a reliable dominant label.
Choose the behaviour supported by most of the sampled frames.
Ignore isolated transition frames.
If no reliable dominant behaviour can be established, choose Unknown.
Do not infer behaviour from the cage, room, lighting or nearby objects.
Answer on one line only:
<Label> {"confidence": <number from 0.0 to 1.0>}
Example: Grooming {"confidence": 0.92}"""

RETRY_SUFFIX_COMPACT = (
    ' You must output exactly one line: one allowed label word '
    '(Eating, Drinking, Grooming, Digging, BellyUp, Active, Resting, None, '
    'Unknown), then one space, then {"confidence": <0.0-1.0>} and nothing else.'
)


def compact_prompt(n_frames=8):
    """Compact prompt; only the frame count is parameterized."""
    p = PROMPT_COMPACT
    if n_frames != 8:
        p = p.replace("These 8 frames", f"These {n_frames} frames")
    return p


def build_window_prompt(n_frames, window_sec, timestamps,
                        sampling=None, target=None, scene=None,
                        template=None):
    """Fill the 9-class spec template with this window's actual values.

    Majority support = floor(n/2)+1 (8 frames -> 5), matching the spec's
    5-of-8 rule generalized to any --frames value.

    template overrides the built-in TEXT_PROMPT (--prompt-file): the same
    placeholder replacements run on the custom text, so a user-edited copy
    of the spec keeps working; a template without placeholders is used
    verbatim."""
    tmpl = template if template is not None else TEXT_PROMPT
    majority = max(1, n_frames // 2 + 1)
    ts = "[" + ", ".join(f"{t:.2f}" for t in timestamps) + "]"
    return (tmpl
            .replace("{NFRAMES}", str(n_frames))
            .replace("{MAJORITY}", str(majority))
            .replace("{DURATION}", f"{window_sec:.2f}")
            .replace("{TIMESTAMPS}", ts)
            .replace("{SAMPLING}", sampling or
                     f"{n_frames} frames sampled uniformly across the window")
            .replace("{TARGET}", target or DEFAULT_TARGET)
            .replace("{SCENE}", scene or DEFAULT_SCENE))


def default_prompt():
    """Filled generic 8-frame prompt (playground default)."""
    return build_window_prompt(8, 6.0, sample_timestamps(0.0, 6.0, 8))


_LOG_PATH_CACHE = {}


def log(tag, msg):
    """Append to logs/vlm_label_<tag>.log with a writable-path fallback
    (sandbox-created files can be locked for user-launched processes)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = _LOG_PATH_CACHE.get(tag, os.path.join(LOG_DIR, f"vlm_label_{tag}.log"))
    for _ in range(3):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            _LOG_PATH_CACHE[tag] = path
            return
        except (PermissionError, OSError):
            path = os.path.join(LOG_DIR,
                                f"vlm_label_{tag}_{int(time.time())}.log")
    print(msg, file=sys.stderr)


def collect_videos(batch_dir, limit=0):
    rels = []
    for root, dirs, files in os.walk(batch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXTS):
                rel = os.path.relpath(os.path.join(root, fn), batch_dir)
                rels.append(rel.replace("\\", "/"))
    rels.sort()
    return rels[:limit] if limit else rels


def window_ranges(duration, window_sec, stride_sec):
    if duration < window_sec:
        return [(0.0, duration)]
    out, t = [], 0.0
    while t + window_sec <= duration + 1e-6:
        out.append((round(t, 3), window_sec))
        t += stride_sec
    if not out:
        out.append((0.0, duration))
    return out


def sample_timestamps(start, length, n_frames):
    return [start + (i + 0.5) * length / n_frames for i in range(n_frames)]


def read_frames_at(path, timestamps):
    """Sequential cv2 read; returns (fps, duration, [BGR frames])."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (n_total / fps) if (n_total > 0 and fps > 0) else 0.0
    ids = [max(0, min(int(round(t * fps)), n_total - 1)) for t in timestamps]
    want = set(ids)
    last_needed = max(ids) if ids else -1
    frames, last, i = {}, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            last = frame
            frames[i] = last
        if i >= last_needed:
            break  # all wanted frames collected; skip decoding the tail
        i += 1
    cap.release()
    out = [frames.get(fid, last) for fid in ids]
    if any(f is None for f in out):
        raise RuntimeError(f"no readable frames in {path}")
    return fps, duration, out


def resize_frames(frames_bgr, max_side):
    import cv2

    out = []
    for f in frames_bgr:
        h, w = f.shape[:2]
        m = max(h, w)
        if m > max_side:
            s = max_side / m
            f = cv2.resize(f, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


def atomic_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    try:
        os.replace(tmp, path)
    except OSError:
        # target file may be locked (sandbox-created); keep data in a twin
        alt = path.replace(".json", f"_{int(time.time())}.json")
        os.replace(tmp, alt)


def merge_windows(raw):
    """[(start,end,label,conf)] -> adjacent-merged segments."""
    segments = []
    for start, end, label, conf in raw:
        if (segments and segments[-1]["label"] == label
                and abs(segments[-1]["end"] - start) < 1e-6):
            prev = segments[-1]
            w0, w1 = prev["_w"], end - start
            prev["confidence"] = (prev["confidence"] * w0 + conf * w1) / (w0 + w1)
            prev["_w"] = w0 + w1
            prev["end"] = end
        else:
            segments.append({"start": start, "end": end, "label": label,
                             "confidence": round(float(conf), 4),
                             "source": "qwen3-vl", "_w": end - start})
    for s in segments:
        s.pop("_w", None)
        s["confidence"] = round(s["confidence"], 4)
    return segments


def parse_answer(text):
    """'<Word> {json...}' -> (label, self_conf, detail or None).

    Resolution order: leading word -> JSON "label" field -> first allowed
    word anywhere in the text (markdown-fence salvage). The leading word
    wins on conflict because it drives the first-token softmax confidence."""
    import re

    label = None
    m = re.match(r"\s*([A-Za-z]+)", text)
    if m and m.group(1).lower() in LABELS:
        label = m.group(1).lower()

    detail, self_conf = None, None
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        try:
            obj = json.loads(text[s:e + 1])
            if isinstance(obj, dict):
                detail = obj
                lab = str(obj.get("label", "")).strip().lower()
                if lab in LABELS and label is None:
                    label = lab
                c = obj.get("confidence")
                if isinstance(c, (int, float)) and not isinstance(c, bool):
                    self_conf = min(1.0, max(0.0, float(c)))
        except ValueError:
            detail = None

    if label is None:
        for w in WORDS:
            if re.search(r"\b" + w + r"\b", text):
                label = w.lower()
                break
    return label, self_conf, detail


def normalize_detail(detail, label, self_conf, n_frames):
    """Coerce the model's JSON detail into stable per-window fields."""
    if not isinstance(detail, dict):
        return None
    detail["label"] = label
    if self_conf is not None:
        detail["confidence"] = round(self_conf, 4)

    def _strs(v):
        return [str(x).strip() for x in v] if isinstance(v, list) else []

    fb = _strs(detail.get("frame_labels"))
    detail["frame_labels"] = fb

    ob = []
    for x in _strs(detail.get("observed_behaviours")):
        xl = x.lower()
        if xl in LABELS and xl not in ("none", "unknown") and xl not in ob:
            ob.append(xl)
    detail["observed_behaviours"] = ob

    try:
        detail["visible_frame_count"] = int(detail.get("visible_frame_count"))
    except (TypeError, ValueError):
        detail["visible_frame_count"] = None
    sfi = []
    for x in detail.get("supporting_frame_indices") or []:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            sfi.append(int(x))
    detail["supporting_frame_indices"] = sfi
    detail["mixed_behaviours"] = bool(detail.get("mixed_behaviours"))
    detail["evidence"] = str(detail.get("evidence", "")).strip()[:300]
    rr = str(detail.get("review_reason", "none")).strip().lower()
    detail["review_reason"] = rr if rr else "none"
    return detail


def load_model(model_id, dtype, gpu_mem_gib, tag):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:  # older/newer transformers fallback
        from transformers import AutoModelForImageTextToText as ModelCls

    log(tag, f"loading {model_id} dtype={dtype} gpu_mem_gib={gpu_mem_gib}")
    t0 = time.time()
    if dtype == "bf16":
        model = ModelCls.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda")
    elif dtype == "bnb4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True)
        max_memory = {0: f"{gpu_mem_gib}GiB"} if gpu_mem_gib > 0 else None
        model = ModelCls.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, quantization_config=bnb,
            device_map="auto", max_memory=max_memory)
    else:
        raise ValueError(f"unknown dtype: {dtype}")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    tok = getattr(processor, "tokenizer", processor)
    cand_ids = []
    for w in WORDS:
        ids = tok.encode(w, add_special_tokens=False)
        cand_ids.append(ids[0] if ids else None)
    use_logits_conf = (all(c is not None for c in cand_ids)
                       and len(set(cand_ids)) == len(WORDS))
    log(tag, f"model ready in {time.time() - t0:.0f}s; "
             f"logits_conf={use_logits_conf} cand_ids={cand_ids}")
    return model, processor, cand_ids, use_logits_conf


def classify_window(model, processor, cand_ids, use_logits_conf, pil_frames,
                    prompt, max_new_tokens, tag, rel, win_idx,
                    retry_suffix=None, detail_mode=True):
    """Returns (label, conf, probs, detail, parse_error). One retry on
    parse failure; final fallback is `unknown` (spec abstention class).

    retry_suffix overrides RETRY_SUFFIX (compact mode uses its own).
    detail_mode=False (compact prompt) skips the 9-field detail JSON: the
    answer only carries {"confidence": x}, stored as a mini-detail so
    windows[].self_confidence still records the self-reported value."""
    import torch

    n_frames = len(pil_frames)
    retry = retry_suffix if retry_suffix is not None else RETRY_SUFFIX

    def build(p):
        messages = [{"role": "user", "content":
                    [{"type": "image", "image": im} for im in pil_frames]
                    + [{"type": "text", "text": p}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        return inputs

    def generate(inputs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        with torch.inference_mode():
            gen = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True)
        return gen

    def first_token_probs(gen):
        if not use_logits_conf:
            return None
        logits = gen.scores[0][0].float()
        cand = torch.tensor(cand_ids, device=logits.device)
        return torch.softmax(logits[cand], dim=-1).cpu().tolist()

    last_err = None
    for attempt, p in enumerate([prompt, prompt + retry]):
        try:
            gen = generate(build(p))
            input_len = gen.sequences.shape[1] - (len(gen.scores))
            text = processor.tokenizer.decode(
                gen.sequences[0][gen.sequences.shape[1] - len(gen.scores):],
                skip_special_tokens=True).strip()
            probs_list = first_token_probs(gen)
            label, self_conf, detail = parse_answer(text)
            probs = None
            if probs_list is not None:
                probs = {lab: round(v, 4) for lab, v in zip(LABELS, probs_list)}
            if label is None:
                last_err = f"unparsed answer: {text[:80]!r}"
                continue
            if detail_mode:
                detail = normalize_detail(detail, label, self_conf, n_frames)
            else:
                detail = ({"confidence": self_conf}
                          if self_conf is not None else None)
            if probs is not None:
                conf = probs[label]
            elif self_conf is not None:
                conf = self_conf
            else:
                conf = 0.5
            return label, round(float(conf), 4), probs, detail, False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                last_err = f"OOM on attempt {attempt}"
                continue
            raise
    log(tag, f"  [WARN] {rel} win{win_idx}: {last_err}")
    return "unknown", 0.0, None, None, True


def crop_frames_yolo(frames_bgr, yolo, expand, conf, tag, rel):
    """Per-frame YOLO crop: detected -> box scaled by `expand` around its
    center (clamped to the frame); miss -> full frame (user-approved rule)."""
    out = []
    n_det = 0
    for f in frames_bgr:
        H, W = f.shape[:2]
        try:
            res = yolo.predict(f, conf=conf, verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                k = int(boxes.conf.argmax())
                x1, y1, x2, y2 = map(float, boxes.xyxy[k].tolist())
                bw, bh = x2 - x1, y2 - y1
                if bw > 2 and bh > 2:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    nw, nh = bw * expand, bh * expand
                    nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
                    nx2, ny2 = min(W, int(cx + nw / 2)), min(H, int(cy + nh / 2))
                    if nx2 - nx1 >= 8 and ny2 - ny1 >= 8:
                        out.append(f[ny1:ny2, nx1:nx2])
                        n_det += 1
                        continue
        except Exception as e:  # noqa: BLE001 - never kill a window on YOLO
            log(tag, f"  [WARN] {rel} yolo frame fail: {e}")
        out.append(f)
    return out, n_det


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True, help="output tag, e.g. qwen3vl-8b-v3")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt", choices=["full", "compact"], default="full",
                    help="full = 9-class spec + JSON detail (schema v3); "
                         "compact = 9-class one-liner, label+confidence "
                         "only (schema v4)")
    ap.add_argument("--prompt-file", default="",
                    help="path to a UTF-8 text file whose full contents "
                         "replace the built-in prompt of the selected "
                         "--prompt mode (prompt experiments). Placeholders "
                         "{NFRAMES}/{MAJORITY}/{DURATION}/{TIMESTAMPS}/"
                         "{SAMPLING}/{TARGET}/{SCENE} are still filled; "
                         "keep the leading label-word answer format so "
                         "parsing and the first-token softmax keep working")
    ap.add_argument("--dtype", choices=["bf16", "bnb4"], default="bf16")
    ap.add_argument("--gpu-mem-gib", type=float, default=0,
                    help="bnb4 only: cap GPU weights (rest offloaded to CPU)")
    ap.add_argument("--window-sec", type=float, default=6.0)
    ap.add_argument("--stride-sec", type=float, default=3.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="drop clips shorter than this many seconds "
                         "(applied before --limit)")
    ap.add_argument("--yolo-crop", action="store_true",
                    help="crop each sampled frame to the YOLO rabbit box "
                         "before inference (miss -> full frame)")
    ap.add_argument("--yolo-weights",
                    default=r"E:\ActionRecognition\detection-src\rabbit-detect"
                            r"\runs\rabbit-yolo11n-0603\weights"
                            r"\rabbit-yolo11n-0603.pt")
    ap.add_argument("--yolo-expand", type=float, default=2.0,
                    help="crop box size = box * expand around its center")
    ap.add_argument("--yolo-conf", type=float, default=0.3)
    a = ap.parse_args(argv)

    import torch
    import numpy as np
    from auto_presort.apply_clips import source_id_for
    from PIL import Image

    tag = a.tag
    custom_prompt = ""
    if a.prompt_file:
        try:
            with open(a.prompt_file, encoding="utf-8") as f:
                custom_prompt = f.read()
        except OSError as e:
            log(tag, f"[ERROR] cannot read prompt file {a.prompt_file}: {e}")
            return 1
        if not custom_prompt.strip():
            log(tag, f"[ERROR] prompt file is empty: {a.prompt_file}")
            return 1
    # custom prompts get a fingerprinted prompt_version so review-page
    # manifests stay distinguishable across experiment rounds
    ph = (hashlib.sha256(custom_prompt.encode("utf-8")).hexdigest()[:8]
          if custom_prompt else "")
    if a.prompt == "compact":
        schema = SCHEMA_VERSION_COMPACT
        pver = f"custom-compact-{ph}" if custom_prompt else PROMPT_VERSION_COMPACT
        retry = RETRY_SUFFIX_COMPACT
        detail_mode = False
        base_prompt = custom_prompt or compact_prompt(a.frames)

        def make_prompt(_length, _ts):
            return base_prompt
    else:
        schema = SCHEMA_VERSION
        pver = f"custom-full-{ph}" if custom_prompt else PROMPT_VERSION
        retry = RETRY_SUFFIX
        detail_mode = True

        def make_prompt(length, ts):
            return build_window_prompt(a.frames, length, ts,
                                       template=custom_prompt or None)
    deadline = float(os.environ.get("VLM_DEADLINE_EPOCH", "0") or 0)
    out_dir = os.path.join(a.dir, f".vlm-{tag}")
    os.makedirs(out_dir, exist_ok=True)

    rels = collect_videos(a.dir, 0)
    if a.min_duration > 0:
        kept = []
        for rel in rels:
            vp = os.path.join(a.dir, *rel.split("/"))
            try:
                _fps, dur, _f = read_frames_at(vp, [0.0])
            except Exception:  # noqa: BLE001
                dur = 0.0
            if dur >= a.min_duration:
                kept.append(rel)
            else:
                log(tag, f"  [SKIP-TINY] {rel} dur={dur:.2f}s < {a.min_duration}")
        rels = kept
    if a.limit:
        rels = rels[:a.limit]
    if not rels:
        log(tag, "[ERROR] no videos found")
        return 1

    if all(os.path.isfile(os.path.join(out_dir, rel.replace("/", "__") + ".json"))
           for rel in rels):
        log(tag, "[SKIP] all clips already labeled; skip model load")
        mp = os.path.join(out_dir, "manifest.json")
        if not os.path.isfile(mp):
            atomic_json(mp, {
                "species": "rabbit",
                "inference_engine": f"qwen3-vl {tag} ({a.dtype})",
                "model_id": a.model_id, "dtype": a.dtype,
                "n_videos": len(rels), "incomplete": False,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 0

    log(tag, f"start dir={a.dir} model={a.model_id} dtype={a.dtype} "
             f"W={a.window_sec}/S={a.stride_sec}/F={a.frames} "
             f"prompt={a.prompt}/{pver}"
             + (f" file={a.prompt_file}" if custom_prompt else "")
             + f" deadline={deadline or 'none'}")
    model, processor, cand_ids, use_logits_conf = load_model(
        a.model_id, a.dtype, a.gpu_mem_gib, tag)

    yolo_model = None
    if a.yolo_crop:
        from ultralytics import YOLO
        yolo_model = YOLO(a.yolo_weights)
        log(tag, f"yolo crop ON: {os.path.basename(a.yolo_weights)} "
                 f"expand={a.yolo_expand} conf={a.yolo_conf}")

    n_windows, n_parsed_err = 0, 0
    label_counts = {l: 0 for l in LABELS}
    incomplete = False
    t_all = time.time()
    for i, rel in enumerate(rels, 1):
        if deadline and time.time() > deadline:
            log(tag, f"[DEADLINE] stopping before clip {i}/{len(rels)}")
            incomplete = True
            break
        flat = rel.replace("/", "__") + ".json"
        out_path = os.path.join(out_dir, flat)
        if os.path.isfile(out_path):
            continue
        t0 = time.time()
        vp = os.path.join(a.dir, *rel.split("/"))
        try:
            fps, duration, _ = read_frames_at(vp, [0.0])
            ranges = window_ranges(duration, a.window_sec, a.stride_sec)
            ts_by_win = [sample_timestamps(s, L, a.frames) for s, L in ranges]
            all_ts = [t for ts in ts_by_win for t in ts]
            _, dur2, flat_frames = read_frames_at(vp, all_ts)
            duration = dur2 or duration

            win_records, raw_windows = [], []
            n_det_total = 0
            for w_idx, (start, length) in enumerate(ranges):
                frames_bgr = flat_frames[w_idx * a.frames:(w_idx + 1) * a.frames]
                if yolo_model is not None:
                    frames_bgr, n_det = crop_frames_yolo(
                        frames_bgr, yolo_model, a.yolo_expand, a.yolo_conf,
                        tag, rel)
                    n_det_total += n_det
                frames_bgr = resize_frames(frames_bgr, a.max_side)
                prompt = make_prompt(length, ts_by_win[w_idx])
                label, conf, probs, detail, perr = classify_window(
                    model, processor, cand_ids, use_logits_conf,
                    [Image.fromarray(np.ascontiguousarray(f[:, :, ::-1]))
                     for f in frames_bgr],
                    prompt, a.max_new_tokens, tag, rel, w_idx,
                    retry_suffix=retry, detail_mode=detail_mode)
                end = duration if w_idx == len(ranges) - 1 else min(start + a.stride_sec, duration)
                raw_windows.append((start, end, label, conf))
                rec = {"start": round(start, 3), "end": round(end, 3),
                       "coverage": round(min(start + length, duration), 3),
                       "label": label, "confidence": conf,
                       "probs": probs, "parse_error": bool(perr)}
                d = detail or {}
                rec.update({
                    "frame_labels": d.get("frame_labels"),
                    "observed_behaviours": d.get("observed_behaviours"),
                    "visible_frame_count": d.get("visible_frame_count"),
                    "supporting_frame_indices": d.get("supporting_frame_indices"),
                    "mixed_behaviours": d.get("mixed_behaviours"),
                    "evidence": d.get("evidence"),
                    "review_reason": d.get("review_reason"),
                    "self_confidence": d.get("confidence"),
                })
                win_records.append(rec)
                n_windows += 1
                label_counts[label] = label_counts.get(label, 0) + 1
                n_parsed_err += int(bool(perr))

            segments = merge_windows(raw_windows)
            meta = {
                "model_id": a.model_id, "dtype": a.dtype,
                "gpu_mem_gib": a.gpu_mem_gib,
                "engine": f"qwen3-vl {tag} ({a.dtype})",
                "prompt_version": pver,
                "window_sec": a.window_sec, "stride_sec": a.stride_sec,
                "frames_per_window": a.frames, "max_side": a.max_side,
                "yolo_crop": bool(a.yolo_crop),
            }
            if a.yolo_crop:
                meta["yolo_weights"] = os.path.basename(a.yolo_weights)
                meta["yolo_expand"] = a.yolo_expand
                meta["yolo_det_ratio"] = round(
                    n_det_total / max(1, len(ranges) * a.frames), 3)
            doc = {
                "schema_version": schema,
                "rel_path": rel,
                "source_id": source_id_for(rel),
                "duration": round(duration, 3),
                "approved": False,
                "segments": segments,
                "gaps": [],
                "windows": win_records,
                "meta": meta,
            }
            atomic_json(out_path, doc)
            log(tag, f"  ({i}/{len(rels)}) {rel} dur={duration:.1f}s "
                     f"wins={len(ranges)} segs={len(segments)} "
                     f"{time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - overnight robustness
            doc = {"schema_version": schema, "rel_path": rel,
                   "source_id": source_id_for(rel), "duration": 0.0,
                   "approved": False, "segments": [], "gaps": [],
                   "windows": [], "error": str(e)[:200],
                   "meta": {"model_id": a.model_id, "dtype": a.dtype,
                            "engine": f"qwen3-vl {tag} ({a.dtype})"}}
            atomic_json(out_path, doc)
            log(tag, f"  ({i}/{len(rels)}) [ERROR] {rel}: {str(e)[:200]}")
        if i % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "species": "rabbit",
        "inference_engine": f"qwen3-vl {tag} ({a.dtype})",
        "model_id": a.model_id, "dtype": a.dtype,
        "gpu_mem_gib": a.gpu_mem_gib,
        "prompt_version": pver,
        "labels": LABELS,
        "params": {"window_sec": a.window_sec, "stride_sec": a.stride_sec,
                   "frames_per_window": a.frames, "max_side": a.max_side,
                   "min_duration": a.min_duration,
                   "max_new_tokens": a.max_new_tokens,
                   "yolo_crop": bool(a.yolo_crop),
                   "yolo_expand": a.yolo_expand if a.yolo_crop else None},
        "n_videos": len(rels), "n_windows": n_windows,
        "label_counts": label_counts, "parse_errors": n_parsed_err,
        "incomplete": incomplete,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    if custom_prompt:
        # keep the exact experiment text next to its results so the review
        # page (and any rerun) can show/reread what was actually used
        manifest["custom_prompt"] = custom_prompt
    atomic_json(os.path.join(out_dir, "manifest.json"), manifest)
    log(tag, f"[DONE] {len(rels)} clips, {n_windows} windows, "
             f"{label_counts}, incomplete={incomplete} "
             f"in {time.time() - t_all:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
