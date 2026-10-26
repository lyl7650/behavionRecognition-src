r"""Prompts + combo configuration for the VLM screener pipeline.

prompt "v3" keeps the user's Unknown class (visible but unreliable/tied ->
Unknown); prompt "old" has its None class removed. Complete rabbit absence
is decided upstream by the YOLO gate (no bbox in a window -> label "none",
the window never reaches the VLM) for BOTH prompts. The pet prompts follow
the same split (v1 = condensed, v2 = legacy wording), both with six classes.

The leading label word feeds the first-token softmax exactly like
scripts.vlm_label (compact mode), so confidence stays comparable.

The label set is per species: rabbit scores five words, pet six. Sharing one
list would let the extra pet word take probability mass away from the rabbit
classes and make old rabbit confidences incomparable with new ones. Callers
pass TABLES[species] down to load_model / classify / parse_answer.
"""

PROJECT_ROOT_HINT = __file__

# rabbit keeps the original five-word list and order (existing results were
# scored against it; do not append to this list)
LABELS = ["eating", "drinking", "active", "resting", "unknown"]
WORDS = ["Eating", "Drinking", "Active", "Resting", "Unknown"]
SCHEMA_VERSION = "vlm-screener-1"
PROMPT_VERSION = {"v3": "screener-5class-v3", "old": "screener-5class-old"}

RETRY_SUFFIX = (
    " You must output exactly one line: one label word "
    "(Eating, Drinking, Active, Resting, Unknown), then one space, then "
    '{"confidence": <0.0-1.0>} and nothing else.'
)

PET_LABELS = LABELS + ["defecating"]
PET_WORDS = WORDS + ["Defecating"]

PET_RETRY_SUFFIX = (
    " You must output exactly one line: one label word "
    "(Eating, Drinking, Active, Resting, Defecating, Unknown), then one "
    'space, then {"confidence": <0.0-1.0>} and nothing else.'
)

# Per-species scoring table. "labels" drives parse_answer and the probability
# zip, "words" drives the first-token candidates, "retry_suffix" is appended
# on a malformed first answer.
TABLES = {
    "rabbit": {"labels": LABELS, "words": WORDS,
               "retry_suffix": RETRY_SUFFIX},
    "pet": {"labels": PET_LABELS, "words": PET_WORDS,
            "retry_suffix": PET_RETRY_SUFFIX},
}

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

# --- pet: both prompts carry the same six classes (Defecating included) ---
# Same label set in both versions on purpose: the four combos vote on one
# window, so a class only one version can emit would always end in a tie.
PET_PROMPT_V1 = """These 8 frames are time-ordered samples from one short clip
of a pet cat or dog. Classify the target animal's dominant
observed state across the clip with exactly one label.
Use these mutually exclusive definitions:
- Eating: the animal is visibly taking, biting or chewing food
in a clear feeding context. Being near food, sniffing a bowl,
lowering the head or chewing a toy is not enough.
Do not confuse eating with grooming.
- Drinking: the animal is actively drinking, with visible lapping
or a clearly supported mouth-water interaction at a bowl,
fountain, bottle or other water source. Being near or looking
at water, or lowering the head over a bowl, is not enough.
- Active: the animal shows clear movement or sustained activity,
including walking, running, jumping, playing, exploring,
digging, grooming or changing body position.
The animal does not need to change location.
Do not use Active when Eating, Drinking or Defecating
is clearly supported. Breathing or an isolated small ear
or tail movement alone is not enough.
- Resting: the animal is sitting quietly, loafing or lying down
without sustained activity. Normal breathing and occasional
small movements are allowed. Do not choose Resting during
grooming, feeding, drinking or sustained exploration.
Do not infer that the animal is asleep.
- Unknown: no target animal is visible, or the animal is visible
but its behaviour cannot be reliably determined because of
occlusion, poor image quality, ambiguous actions or no clear
dominant state. Do not force an uncertain behaviour into
another label.
- Defecating: the animal is visibly passing faeces, or the frame
sequence clearly supports a defecation event through a sustained
elimination posture together with visible faeces emerging or
appearing directly beneath or behind the animal.
Squatting, crouching, entering a litter box, digging, covering
litter or remaining in a toilet area alone is not enough.
Do not confuse urination with defecation. If elimination is
apparent but defecation cannot be distinguished from urination,
choose Unknown.
Choose the behaviour supported by most of the sampled frames.
Ignore isolated transition frames.
Use a specific behaviour instead of Active or Resting only
where that behaviour is clearly supported.
If behaviours are tied or no reliable dominant state can be
established, choose Unknown.
Answer on one line only, using exactly one of:
Eating, Drinking, Active, Resting, Unknown, Defecating.
Format:
<Label> {"confidence": <number from 0.0 to 1.0>}
Example: Eating {"confidence": 0.92}"""

PET_PROMPT_V2 = """These 8 frames are consecutive time-ordered samples from one short clip of a pet dog or cat. Classify the animal's dominant state across the clip with exactly one label. Use these mutually exclusive definitions:
- Eating: the animal is visibly taking, biting or chewing food in a clear feeding context. Typical evidence is food visible in the mouth, the mouth on the food, or a food source directly in use. Do not choose Eating just because the head is lowered, the animal is sniffing a bowl or the floor, or it is chewing a toy or a non-food object. Do not confuse eating with grooming.
- Drinking: the animal is actively drinking, with visible lapping or a clearly supported mouth-water interaction at a bowl, fountain, bottle or other water source. Being near or looking at the water source is not enough. Lowering the head over a bowl without visible lapping is not enough.
- Active: the animal shows any movement or sustained non-stationary behavior, including walking, running, jumping, playing, exploring, digging, standing and looking around, or grooming and licking its fur in place. The animal does not need to change location to be Active. Do not use Active when Eating, Drinking or Defecating is clearly supported. Breathing or an isolated ear or tail movement alone is not enough.
- Resting: the animal is sitting quietly, loafing, lying down or curled up and its body is essentially still. Normal breathing and occasional small movements are allowed. Do not choose Resting for an animal that is grooming, feeding, drinking or moving around. Do not infer that the animal is asleep.
- Defecating: the animal is visibly passing faeces, or the frame sequence clearly supports a defecation event through a sustained elimination posture together with visible faeces emerging or appearing directly beneath or behind the animal. Squatting, crouching, entering a litter box, digging, covering litter or remaining in a toilet area alone is not enough. Do not confuse urination with defecation. If elimination is apparent but defecation cannot be distinguished from urination, choose Unknown.
- Unknown: the animal is visible in at least one frame, but occlusion, poor image quality, ambiguous behavior or an unresolvable tie prevents a reliable dominant label. Never choose Unknown when Eating, Drinking, Active, Resting or Defecating is clearly supported for most of the 8 frames.
When more than one behavior appears, choose the behavior that is visible for most of the 8 frames and ignore isolated transition frames. If the behaviors are tied, prefer Eating or Drinking, then Active, then Resting.
Answer in EXACTLY this format on one line and nothing else:
Eating|Drinking|Active|Resting|Defecating|Unknown {"confidence": 0.0-1.0}
Example: Eating {"confidence": 0.92}"""

PET_PROMPTS = {"v1": PET_PROMPT_V1, "v2": PET_PROMPT_V2}
PET_PROMPT_VERSION = {"v1": "screener-pet-6class-v1",
                      "v2": "screener-pet-6class-v2"}

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
