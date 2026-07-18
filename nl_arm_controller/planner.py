"""LLM motion planner — DeepSeek turns natural language into a validated `MotionPlan`.

Architecture:
  utterance ──► DeepSeek (constrained JSON) ──► parse_plan_json ──► MotionPlan
                                                       │
                                                       ▼
                                              validate_plan (Phase 3)
                                                       │
                                                       ▼
                                              MotionExecutor.execute

Requires:
  pip install openai>=1.0.0

Environment variables:
  DEEPSEEK_API_KEY   — your DeepSeek API key
  DEEPSEEK_MODEL     — optional override (default: deepseek-chat)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from motion import MotionPlan, validate_plan


SYSTEM_PROMPT = """You are the motion planner for an SO-101 6-axis robot arm.

You translate the user's natural-language request into a JSON motion plan that
will drive the arm. You do nothing else — no chat, no apologies, no markdown.

The arm has 6 revolute joints, named "1" through "6":
  "1" — base rotation         (±90°)   positive = turn RIGHT (clockwise from above)
  "2" — shoulder pitch        (±60°)   positive = lift arm UP
  "3" — elbow                 (±60°)   positive = extend FORWARD
  "4" — wrist pitch           (±60°)   positive = wrist UP
  "5" — wrist roll            (±60°)   positive = roll CW
  "6" — gripper / wrist yaw   (±60°)

Output format — return EXACTLY one JSON object, no code fences, no commentary:

{
  "say": "<one short sentence describing the motion>",
  "actions": [
    { "type": "set_joint", "joint": "1", "angle": 30, "duration": 1.5 },
    { "type": "set_pose",  "pose": {"1": 30, "2": -20}, "duration": 2.0 },
    { "type": "wait",      "duration": 0.5 },
    { "type": "home",      "duration": 1.5 }
  ]
}

Rules:
- "actions" must contain 1–8 entries.
- Every "duration" is seconds, range 0.0–5.0 per action.
- "angle" is degrees. Stay within the per-joint range above. ±20°–45° looks expressive.
- Use "set_joint" for single-joint moves, "set_pose" for coordinated multi-joint moves,
  "wait" for pauses, "home" to return every joint to 0°.
- ALWAYS finish with a return-toward-zero (either "home" or a final move to 0°)
  unless the user explicitly asks the arm to hold a pose.
- If the request is unsafe, ambiguous, or impossible, still return a valid plan —
  pick a small conservative gesture and explain it in "say".
- NEVER output prose outside the JSON object.

Few-shot examples:

User: "wave at the audience"
{"say":"Waving from my base.","actions":[{"type":"set_joint","joint":"1","angle":30,"duration":0.7},{"type":"set_joint","joint":"1","angle":-30,"duration":1.0},{"type":"set_joint","joint":"1","angle":30,"duration":1.0},{"type":"set_joint","joint":"1","angle":0,"duration":0.7}]}

User: "look up and to the right"
{"say":"Tilting up and turning right.","actions":[{"type":"set_pose","pose":{"1":25,"2":-25},"duration":1.5},{"type":"wait","duration":0.5},{"type":"home","duration":1.5}]}

User: "stop"
{"say":"Stopping and going home.","actions":[{"type":"home","duration":1.0}]}
"""


@dataclass
class PlanResult:
    """Outcome of a single planner call."""

    plan: MotionPlan | None
    raw_response: str
    error: str | None
    model: str

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None


_FENCE_HEAD = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_TAIL = re.compile(r"\s*```$")


def parse_plan_json(raw: str) -> tuple[MotionPlan | None, str | None]:
    """Best-effort extract a JSON object from `raw` and convert to MotionPlan.

    Returns (plan, None) on success, (None, error_message) on failure.
    """
    text = (raw or "").strip()
    if not text:
        return None, "empty response"

    text = _FENCE_HEAD.sub("", text)
    text = _FENCE_TAIL.sub("", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, f"no JSON object found in response: {raw[:200]!r}"

    blob = text[start : end + 1]

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}  (blob: {blob[:200]!r})"

    if not isinstance(data, dict):
        return None, f"top-level JSON must be an object, got {type(data).__name__}"

    try:
        plan = MotionPlan.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"plan shape error: {exc}  (data: {data!r})"

    errors = validate_plan(plan)
    if errors:
        return None, "validation failed:\n  - " + "\n  - ".join(errors)

    return plan, None


def _make_client():
    """Create an OpenAI-compatible client pointed at DeepSeek."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Get your key at https://platform.deepseek.com"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def plan_from_utterance(
    utterance: str,
    *,
    model: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> PlanResult:
    """Call DeepSeek with `utterance` and return a `PlanResult`.

    Picks up DEEPSEEK_API_KEY from the environment.
    """
    client = _make_client()
    chosen_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    response = client.chat.completions.create(
        model=chosen_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ],
    )

    raw = response.choices[0].message.content or ""

    plan, err = parse_plan_json(raw)
    return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)


# ---------------------------------------------------------------------------
# Vision-aware planning
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are the motion planner AND scene narrator for an SO-101 6-axis robot arm.
You receive an image from the arm's workspace camera AND a natural-language
request from the operator. Your job is to return a single JSON object that
either describes what you see, plans an arm motion, or does both.

The arm has 6 revolute joints, named "1" through "6":
  "1" — base rotation         (±90°)   positive = turn RIGHT (clockwise from above)
  "2" — shoulder pitch        (±60°)   positive = lift arm UP
  "3" — elbow                 (±60°)   positive = extend FORWARD
  "4" — wrist pitch           (±60°)   positive = wrist UP
  "5" — wrist roll            (±60°)   positive = roll CW
  "6" — gripper / wrist yaw   (±60°)

The camera is mounted near the operator looking at the arm's workspace.
"left" / "right" in your descriptions refer to the audience's view, which
matches the camera frame.

Output format — return EXACTLY one JSON object, no code fences, no markdown,
no commentary outside the JSON. Schema:

{
  "say":     "<the spoken response — describe the scene, answer the question,\
 or narrate the motion>",
  "actions": [ ... 0 to 8 motion actions, same shape as the text-only planner ... ]
}

Action shapes (any combination, in order):
  { "type": "set_joint", "joint": "1"-"6", "angle": -90..90, "duration": 0.1..5.0 }
  { "type": "set_pose",  "pose": {"1": deg, "2": deg, ...},  "duration": 0.1..5.0 }
  { "type": "wait",                                         "duration": 0.1..5.0 }
  { "type": "home",                                         "duration": 0.1..5.0 }

Decision rules:

1. If the operator asks ABOUT the scene ("what do you see?", "is there a red
   cup?", "where's my notebook?"), respond ONLY with description in `say` and
   an empty `actions` array. NO MOTION for purely informational questions.

2. If the operator asks for MOTION ("wave at the audience", "do a small bow"),
   plan motion in `actions`. The `say` should briefly narrate what you'll do.

3. If the operator asks for VISUALLY-GROUNDED MOTION ("wave at the red cup",
   "look at the laptop"), describe what you see in `say`, then plan a small
   gesture toward the relevant area. Without precise camera-arm calibration
   you cannot point exactly — aim with joint "1" (base rotation) toward the
   approximate horizontal direction (left = negative, right = positive,
   typically ±20° to ±40°).

4. If the operator references something you DON'T see, say so honestly and
   leave `actions` empty. Do not pretend or hallucinate motion.

5. ALWAYS return to a near-zero pose at the end of any motion (either via a
   final "home" action or a final move to 0°) unless the operator explicitly
   says to hold a pose.

6. NEVER output prose outside the JSON object. NEVER use code fences.

Few-shot examples:

User says: "what's on the table?"
{"say":"I see a red cup on the right side, a laptop in the middle, and a blue notebook to the left.","actions":[]}

User says: "wave at the audience"
{"say":"Waving from my base.","actions":[{"type":"set_joint","joint":"1","angle":30,"duration":0.7},{"type":"set_joint","joint":"1","angle":-30,"duration":1.0},{"type":"set_joint","joint":"1","angle":30,"duration":1.0},{"type":"set_joint","joint":"1","angle":0,"duration":0.7}]}

User says: "look at the red cup"
{"say":"I see a red cup on the right. Turning toward it.","actions":[{"type":"set_pose","pose":{"1":35,"2":-15},"duration":1.5},{"type":"wait","duration":0.5},{"type":"home","duration":1.5}]}

User says: "do you see a banana?"
{"say":"No, I don't see a banana — I see a red cup, a laptop, and a notebook.","actions":[]}
"""


# Vision-capable DeepSeek model — update if DeepSeek releases a newer one
_DEEPSEEK_VISION_MODEL = "deepseek-chat"  # deepseek-chat supports vision as of V3


def plan_from_utterance_with_image(
    utterance: str,
    frame_b64_jpeg: str | None,
    *,
    model: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> PlanResult:
    """Call DeepSeek Vision with `utterance` + the image, return a `PlanResult`.

    If `frame_b64_jpeg` is None (no fresh frame available), falls back to the
    text-only planner so the agent stays usable when the camera publisher is down.
    """
    if frame_b64_jpeg is None:
        return plan_from_utterance(
            utterance, model=model, max_tokens=max_tokens, temperature=temperature
        )

    client = _make_client()
    chosen_model = model or os.environ.get("DEEPSEEK_MODEL", _DEEPSEEK_VISION_MODEL)

    response = client.chat.completions.create(
        model=chosen_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64_jpeg}",
                        },
                    },
                    {"type": "text", "text": utterance},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content or ""

    plan, err = parse_plan_json(raw)
    return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)