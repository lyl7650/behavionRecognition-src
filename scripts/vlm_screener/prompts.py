r"""Prompts + combo configuration for the VLM screener pipeline.

prompt "v3" keeps the user's Unknown class (visible but unreliable/tied ->
Unknown); prompt "old" has its None class removed. Complete rabbit absence
is decided upstream by the YOLO gate (no bbox in a window -> label "none",
the window never reaches the VLM) for BOTH prompts.

The leading label word feeds the first-token softmax exactly like
scripts.vlm_label (compact mode), so confidence stays comparable.
"""

PROJECT_ROOT_HINT = __file__

LABELS = ["eating", "drinking", "active", "resting", "unknown"]
WORDS = ["Eating", "Drinking", "Active", "Resting", "Unknown"]
SCHEMA_VERSION = "vlm-screener-1"
PROMPT_VERSION = {"v3": "screener-5class-v3", "old": "screener-5class-old"}

RETRY_SUFFIX = (
    " You must output exactly one line: one label word "
    "(Eating, Drinking, Active, Resting, Unknown), then one space, then "
    '{"confidence": <0.0-1.0>} and nothing else.'
)

# --- prompt "v3": user's condensed spec, verbatim (Unknown kept) ----------
PROMPT_V3 = """These 8 frames are time-ordered samples from one short clip
of a pet rabbit. Classify the rabbit's dominant observed state
with exactly one label.
Use these definitions:
- Eating: visibly taking or chewing identifiable food, including
hay, grass or pellets. Head-down posture or proximity to food
alone is not enough. Do not confuse eating with grooming.
- Drinking: actively drinking at a water bowl or bottle.
Being near a water source is not enough.
- Active: clear movement or sustained activity, including hopping,
running, rearing, exploring or rolling. Do not use Active for
Eating, Drinking, Grooming or Digging.
- Resting: sitting quietly, loafing or lying down without sustained
activity. Breathing and small occasional movements are allowed.
- Unknown: a rabbit is visible, but poor visibility, ambiguous
behaviour or a tie prevents a reliable dominant label.
Choose the behaviour supported by most of the sampled frames.
Ignore isolated transition frames.
If no reliable dominant behaviour can be established, choose Unknown.
Do not infer behaviour from the cage, room, lighting or nearby objects.
Answer on one line only:
<Label> {"confidence": <number from 0.0 to 1.0>}
Example: Eating {"confidence": 0.92}"""

# --- prompt "old": user's legacy spec, None replaced by Unknown -----------
# (complete rabbit absence is handled by the YOLO gate, not the prompt)
PROMPT_OLD = """These 8 frames are consecutive time-ordered samples from one short clip of a pet rabbit. Classify the rabbit's dominant state across the clip with exactly one label. Use these mutually exclusive definitions:
- Eating: the rabbit is eating grass or hay. Typical evidence is the head lowered to the grass/hay or the mouth visibly chewing grass/hay. Do not choose Eating just because the head is down, the rabbit is sniffing, or the rabbit is grooming.
- Drinking: the rabbit is actively drinking with its mouth at a water bowl or water bottle. Being near or looking at the water source is not enough.
- Active: the rabbit shows any movement or non-stationary behavior, including hopping, running, fast movement, rearing, digging, standing and looking around, or grooming/licking its fur in place. The rabbit does not need to change location to be Active.
- Resting: the rabbit is lying down or fully flattened and its body is completely still. Do not choose Resting for an upright rabbit that is looking around, grooming, or otherwise moving.
- Unknown: a rabbit is visible in at least one frame, but poor visibility, ambiguous behaviour or an unresolvable tie prevents a reliable dominant label. Never choose Unknown when Eating, Drinking, Active or Resting is clearly supported for most of the 8 frames.
When more than one behavior appears, choose the behavior that is visible for most of the 8 frames and ignore isolated transition frames. If the behaviors are tied, prefer Eating or Drinking, then Active, then Resting. Resting still requires complete stillness while lying down or flattened.
Answer in EXACTLY this format on one line and nothing else:
Resting|Active|Eating|Drinking|Unknown {"confidence": 0.0-1.0}
Example: Eating {"confidence": 0.92}"""

PROMPTS = {"v3": PROMPT_V3, "old": PROMPT_OLD}

# Supplied later; pet inference is disabled until both texts are set.
PET_PROMPT_V1 = None
PET_PROMPT_V2 = None
PET_PROMPTS = {"v1": PET_PROMPT_V1, "v2": PET_PROMPT_V2}
PET_PROMPT_VERSION = {"v1": "screener-pet-5class-v1",
                      "v2": "screener-pet-5class-v2"}

# combo = model x prompt. Order = execution order (8B first, cheap).
COMBOS = [
    {"combo": "8b-v3",   "model_key": "8b",  "dtype": "bf16", "prompt_key": "v3"},
    {"combo": "8b-old",  "model_key": "8b",  "dtype": "bf16", "prompt_key": "old"},
    {"combo": "32b-v3",  "model_key": "32b", "dtype": "bnb4", "prompt_key": "v3"},
    {"combo": "32b-old", "model_key": "32b", "dtype": "bnb4", "prompt_key": "old"},
]
PET_COMBOS = [
    {"combo": f"{model}-{version}", "model_key": model,
     "dtype": dtype, "prompt_key": version}
    for model, dtype in (("8b", "bf16"), ("32b", "bnb4"))
    for version in ("v1", "v2")
]
COMBOS_BY_SPECIES = {"rabbit": COMBOS, "pet": PET_COMBOS}
PROMPTS_BY_SPECIES = {"rabbit": PROMPTS, "pet": PET_PROMPTS}
VERSIONS_BY_SPECIES = {"rabbit": PROMPT_VERSION, "pet": PET_PROMPT_VERSION}

MODEL_KEYS = {"8b": "qwen3-vl-8b", "32b": "qwen3-vl-32b"}

# YOLO gate: a window counts as containing the rabbit when at least
# GATE_MIN_DET of its GATE_N_FRAMES frames have a detection (conf >= 0.3).
GATE_MIN_DET = 4
YOLO_CONF = 0.3

# compact one-liner answers need ~20 tokens; 64 leaves retry head-room
MAX_NEW_TOKENS = 64
