"""server.py — Makeup Artist Robot WebSocket server.

Runs the arm motion and streams joint angles to the browser in real-time.
The browser maps those angles onto the uploaded face image.

Usage:
    python server.py            # connects to Cyberwave twin
    python server.py --dry-run  # no robot, still streams fake angles for UI testing

Architecture:
    Browser  ──WS──►  server.py  ──►  MotionExecutor  ──►  Cyberwave twin
                          │
                          ◄── joint angle stream (real-time)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(override=False)

import websockets
from websockets.server import WebSocketServerProtocol

from motion import Action, MotionExecutor, MotionPlan


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST = "localhost"
PORT = 8765

CYBERWAVE_API_KEY = os.environ.get("CYBERWAVE_API_KEY")
CW_MODE          = os.environ.get("CW_MODE", "simulation")
CW_TWIN_ID       = os.environ.get("CYBERWAVE_TWIN_ID")
CW_ENV_ID        = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
TWIN_ASSET_KEY   = "the-robot-studio/so101"


# ---------------------------------------------------------------------------
# Makeup action registry
# Each entry defines:
#   plan      — the MotionPlan to execute
#   region    — which facial landmark region this action targets
#   joints    — which joints drive the canvas stroke (horizontal, vertical)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lipstick motion tuning
# ---------------------------------------------------------------------------
# ARM POSITION
LIFT_HEIGHT     = -28.0   # _2: shoulder lift toward face height (more negative = higher)
REACH_FORWARD   =  18.0   # _3: elbow extension toward face (bigger = further forward)
WRIST_TILT_DOWN =  -8.0   # _4: wrist tilts down at centre of lips (negative = tip down)
WRIST_TILT_FLAT =   0.0   # _4: wrist levels out at lip corners

# STROKE ANGLES — how far _1 sweeps left/right
CENTRE_ANGLE    =   0.0   # _1: centre of lips
LEFT_CORNER     =  13.0   # _1: left corner of lips  (positive = arm swings left)
RIGHT_CORNER    = -13.0   # _1: right corner of lips (negative = arm swings right)

# STROKE SPEEDS — duration in seconds per segment
APPROACH_DUR    =  1.2    # time to lift and extend toward face
CENTRE_PAUSE    =  0.35   # brief press at centre before stroking
UPPER_SWEEP_DUR =  0.50   # duration of each upper lip segment (centre→corner)
LOWER_SWEEP_DUR =  0.65   # duration of each lower lip sweep (fuller, slower)
BLOT_OUT_DUR    =  0.18   # quick forward press for blot
BLOT_IN_DUR     =  0.18   # quick pullback for blot
RETREAT_DUR     =  1.5    # time to return home

# REPETITIONS — how many times each phase repeats
# Set to 1 for a single clean pass; 2–3 for more coverage
UPPER_LIP_REPS  =  2      # number of full upper lip passes (centre→left→centre→right)
LOWER_LIP_REPS  =  2      # number of full lower lip sweeps (left→right)
BLOT_REPS       =  1      # number of blot dabs at the end


def _lipstick_plan() -> MotionPlan:
    """Natural lipstick application sequence:
      1. Approach  — lift arm to face height and extend forward
      2. Upper lip — centre → left corner → centre → right corner (×UPPER_LIP_REPS)
      3. Lower lip — left → right full sweep (×LOWER_LIP_REPS)
      4. Blot      — small forward dab to set the colour (×BLOT_REPS)
      5. Retreat   — smooth return to home
    """
    actions: list[Action] = []

    # ── 1. Approach ──────────────────────────────────────────────
    actions.append(Action(
        type="set_pose",
        pose={"_2": LIFT_HEIGHT, "_3": REACH_FORWARD, "_4": WRIST_TILT_DOWN},
        duration=APPROACH_DUR,
    ))
    actions.append(Action(type="wait", duration=CENTRE_PAUSE))

    # ── 2. Upper lip passes ───────────────────────────────────────
    # Traces the cupid's bow: centre → left corner → centre → right corner
    # _1 sweeps horizontally; _4 levels out at corners to follow the bow curve
    for _ in range(UPPER_LIP_REPS):
        # Centre → left corner (wrist levels as arm swings out)
        actions.append(Action(
            type="set_pose",
            pose={"_1": LEFT_CORNER, "_4": WRIST_TILT_FLAT},
            duration=UPPER_SWEEP_DUR,
        ))
        # Left corner → centre (wrist dips back down at centre)
        actions.append(Action(
            type="set_pose",
            pose={"_1": CENTRE_ANGLE, "_4": WRIST_TILT_DOWN},
            duration=UPPER_SWEEP_DUR,
        ))
        # Centre → right corner
        actions.append(Action(
            type="set_pose",
            pose={"_1": RIGHT_CORNER, "_4": WRIST_TILT_FLAT},
            duration=UPPER_SWEEP_DUR,
        ))
        # Right corner → centre
        actions.append(Action(
            type="set_pose",
            pose={"_1": CENTRE_ANGLE, "_4": WRIST_TILT_DOWN},
            duration=UPPER_SWEEP_DUR,
        ))

    # ── 3. Lower lip sweeps ───────────────────────────────────────
    # Fuller, slower sweep across the lower lip
    # _4 tilts slightly upward for lower lip (opposite of upper)
    for _ in range(LOWER_LIP_REPS):
        # Sweep left to right
        actions.append(Action(
            type="set_pose",
            pose={"_1": LEFT_CORNER, "_4": 5.0},
            duration=LOWER_SWEEP_DUR,
        ))
        actions.append(Action(
            type="set_pose",
            pose={"_1": RIGHT_CORNER, "_4": 5.0},
            duration=LOWER_SWEEP_DUR,
        ))

    # Return to centre after lower lip
    actions.append(Action(
        type="set_pose",
        pose={"_1": CENTRE_ANGLE, "_4": WRIST_TILT_DOWN},
        duration=UPPER_SWEEP_DUR,
    ))

    # ── 4. Blot ───────────────────────────────────────────────────
    # Small forward press to set the colour, like pressing lips together
    for _ in range(BLOT_REPS):
        actions.append(Action(
            type="set_joint", joint="_3",
            angle=REACH_FORWARD + 5.0,
            duration=BLOT_OUT_DUR,
        ))
        actions.append(Action(
            type="set_joint", joint="_3",
            angle=REACH_FORWARD,
            duration=BLOT_IN_DUR,
        ))

    # ── 5. Retreat ────────────────────────────────────────────────
    actions.append(Action(type="home", duration=RETREAT_DUR))

    return MotionPlan(say="Applying lipstick.", actions=actions)


# ---------------------------------------------------------------------------
# Eyebrow motion tuning
# ---------------------------------------------------------------------------
# ARM POSITION
BROW_LIFT_HEIGHT    = -42.0   # _2: shoulder height at brow level
BROW_REACH_FORWARD  =  12.0   # _3: elbow extension toward face

# STROKE POSITIONS — _1 angle for each brow's inner/outer boundary
BROW_L_INNER        =   8.0   # left brow inner corner
BROW_L_OUTER        =  26.0   # left brow outer corner
BROW_R_INNER        =  -8.0   # right brow inner corner
BROW_R_OUTER        = -26.0   # right brow outer corner

# LATERAL PASS tuning
# Each pass is one continuous inner→outer sweep at a slightly different height.
# Passes stack vertically to build up the layered hair texture of the brow.
BROW_PASS_COUNT     =   5     # number of lateral passes per brow
                               # more passes = fuller/denser brow
BROW_PASS_DUR       =  0.55   # duration of each inner→outer sweep
BROW_RETURN_DUR     =  0.30   # duration of outer→inner return between passes
BROW_V_STEP         =   1.5   # _2 shift per pass (degrees) — each pass sits
                               # slightly higher, building the brow shape
                               # positive = passes move upward on the brow
BROW_WRIST_INNER    =  -3.0   # _4 at inner corner (wrist angled for brow base)
BROW_WRIST_OUTER    =   4.0   # _4 at outer corner (follows brow tail angle)

# GENERAL TIMING
BROW_APPROACH_DUR   =  1.2    # time to lift arm to brow height
BROW_PAUSE_DUR      =  0.2    # pause at start before stroking
BROW_RETREAT_DUR    =  1.5    # time to return home
BROW_BETWEEN_DUR    =  0.9    # reposition time between left and right brow


def _brow_passes_one_side(
    inner: float, outer: float, lift_base: float
) -> list[tuple[Action, dict]]:
    """Generate BROW_PASS_COUNT lateral passes across one eyebrow.
    Returns list of (Action, metadata) tuples so the browser knows
    which pass is a sweep vs a return.
    """
    result: list[tuple[Action, dict]] = []

    for i in range(BROW_PASS_COUNT):
        v_offset = i * BROW_V_STEP

        # Sweep: inner → outer (paint this)
        result.append((
            Action(
                type="set_pose",
                pose={"_1": outer, "_2": lift_base + v_offset, "_4": BROW_WRIST_OUTER},
                duration=BROW_PASS_DUR,
            ),
            {"brow_phase": "sweep", "brow_pass": i, "brow_total": BROW_PASS_COUNT},
        ))

        # Return: outer → inner (skip painting)
        if i < BROW_PASS_COUNT - 1:
            result.append((
                Action(
                    type="set_pose",
                    pose={"_1": inner, "_2": lift_base + v_offset, "_4": BROW_WRIST_INNER},
                    duration=BROW_RETURN_DUR,
                ),
                {"brow_phase": "return", "brow_pass": i, "brow_total": BROW_PASS_COUNT},
            ))

    return result


def _build_brow_plan(
    say: str,
    sides: list[tuple[float, float]],   # [(inner, outer), ...]
) -> MotionPlan:
    """Build a brow plan with phase metadata attached as action tags."""
    tagged: list[tuple[Action, dict]] = []

    for side_idx, (inner, outer) in enumerate(sides):
        # Approach
        tagged.append((
            Action(
                type="set_pose",
                pose={"_1": inner, "_2": BROW_LIFT_HEIGHT,
                      "_3": BROW_REACH_FORWARD, "_4": BROW_WRIST_INNER},
                duration=BROW_APPROACH_DUR,
            ),
            {"brow_phase": "approach"},
        ))
        tagged.append((
            Action(type="wait", duration=BROW_PAUSE_DUR),
            {"brow_phase": "approach"},
        ))

        # Passes
        tagged.extend(_brow_passes_one_side(inner, outer, BROW_LIFT_HEIGHT))

        # Between sides reposition (if more sides follow)
        if side_idx < len(sides) - 1:
            next_inner = sides[side_idx + 1][0]
            tagged.append((
                Action(
                    type="set_pose",
                    pose={"_1": next_inner, "_2": BROW_LIFT_HEIGHT, "_4": BROW_WRIST_INNER},
                    duration=BROW_BETWEEN_DUR,
                ),
                {"brow_phase": "reposition"},
            ))
            tagged.append((
                Action(type="wait", duration=BROW_PAUSE_DUR),
                {"brow_phase": "reposition"},
            ))

    # Retreat
    tagged.append((
        Action(type="home", duration=BROW_RETREAT_DUR),
        {"brow_phase": "retreat"},
    ))

    # Store metadata on plan so StreamingExecutor can forward it
    plan = MotionPlan(say=say, actions=[a for a, _ in tagged])
    plan._action_meta = [m for _, m in tagged]   # type: ignore[attr-defined]
    return plan


def _left_eyebrow_plan() -> MotionPlan:
    return _build_brow_plan(
        "Drawing left eyebrow.",
        [(BROW_L_INNER, BROW_L_OUTER)],
    )


def _right_eyebrow_plan() -> MotionPlan:
    return _build_brow_plan(
        "Drawing right eyebrow.",
        [(BROW_R_INNER, BROW_R_OUTER)],
    )


def _both_eyebrows_plan() -> MotionPlan:
    return _build_brow_plan(
        "Drawing both eyebrows.",
        [(BROW_L_INNER, BROW_L_OUTER), (BROW_R_INNER, BROW_R_OUTER)],
    )


# ---------------------------------------------------------------------------
# Blush motion tuning
# ---------------------------------------------------------------------------
# ARM POSITION
CHEEK_LIFT_HEIGHT   = -20.0   # _2: shoulder height at cheek level (lower on the face than the brows)
CHEEK_REACH_FORWARD =  14.0   # _3: elbow extension toward face

# BUFFING POSITIONS — _1 angle at the centre of each cheek, and how far the
# arm wobbles side-to-side while "buffing" the blush in
CHEEK_L_CENTER      =  34.0   # left cheek centre
CHEEK_R_CENTER      = -34.0   # right cheek centre
CHEEK_WOBBLE        =   5.0   # +/- degrees swept on each buffing pass

# BUFFING PASS tuning
# Each pass is one small side-to-side buff at the cheek. Passes stack to
# build up soft, blended colour density, like real blush application.
CHEEK_PASS_COUNT    =   4     # number of buffing passes per cheek
                               # more passes = deeper/denser blush
CHEEK_PASS_DUR      =  0.45   # duration of each buffing sweep
CHEEK_RETURN_DUR    =  0.30   # duration of the return between passes
CHEEK_WRIST_TILT    =   6.0   # _4: gentle downward-angled dab, like a brush

# GENERAL TIMING
CHEEK_APPROACH_DUR  =  1.1    # time to lift arm to cheek height
CHEEK_PAUSE_DUR     =  0.2    # pause before buffing starts
CHEEK_RETREAT_DUR   =  1.5    # time to return home
CHEEK_BETWEEN_DUR   =  0.9    # reposition time between left and right cheek


def _cheek_passes_one_side(centre: float) -> list[tuple[Action, dict]]:
    """Generate CHEEK_PASS_COUNT small buffing passes across one cheek.
    Returns list of (Action, metadata) tuples so the browser knows which
    pass is a sweep (paint) vs a return (don't paint).
    """
    result: list[tuple[Action, dict]] = []
    lo = centre - CHEEK_WOBBLE
    hi = centre + CHEEK_WOBBLE

    for i in range(CHEEK_PASS_COUNT):
        # Sweep: lo → hi (paint this — builds up the blush)
        result.append((
            Action(
                type="set_pose",
                pose={"_1": hi, "_2": CHEEK_LIFT_HEIGHT, "_4": CHEEK_WRIST_TILT},
                duration=CHEEK_PASS_DUR,
            ),
            {"cheek_phase": "sweep", "cheek_pass": i, "cheek_total": CHEEK_PASS_COUNT},
        ))

        # Return: hi → lo (skip painting)
        if i < CHEEK_PASS_COUNT - 1:
            result.append((
                Action(
                    type="set_pose",
                    pose={"_1": lo, "_2": CHEEK_LIFT_HEIGHT, "_4": CHEEK_WRIST_TILT},
                    duration=CHEEK_RETURN_DUR,
                ),
                {"cheek_phase": "return", "cheek_pass": i, "cheek_total": CHEEK_PASS_COUNT},
            ))

    return result


def _build_cheek_plan(say: str, centres: list[float]) -> MotionPlan:
    """Build a blush plan with phase metadata attached as action tags."""
    tagged: list[tuple[Action, dict]] = []

    for side_idx, centre in enumerate(centres):
        lo = centre - CHEEK_WOBBLE

        # Approach
        tagged.append((
            Action(
                type="set_pose",
                pose={"_1": lo, "_2": CHEEK_LIFT_HEIGHT,
                      "_3": CHEEK_REACH_FORWARD, "_4": CHEEK_WRIST_TILT},
                duration=CHEEK_APPROACH_DUR,
            ),
            {"cheek_phase": "approach"},
        ))
        tagged.append((
            Action(type="wait", duration=CHEEK_PAUSE_DUR),
            {"cheek_phase": "approach"},
        ))

        # Passes
        tagged.extend(_cheek_passes_one_side(centre))

        # Between sides reposition (if more sides follow)
        if side_idx < len(centres) - 1:
            next_lo = centres[side_idx + 1] - CHEEK_WOBBLE
            tagged.append((
                Action(
                    type="set_pose",
                    pose={"_1": next_lo, "_2": CHEEK_LIFT_HEIGHT, "_4": CHEEK_WRIST_TILT},
                    duration=CHEEK_BETWEEN_DUR,
                ),
                {"cheek_phase": "reposition"},
            ))
            tagged.append((
                Action(type="wait", duration=CHEEK_PAUSE_DUR),
                {"cheek_phase": "reposition"},
            ))

    # Retreat
    tagged.append((
        Action(type="home", duration=CHEEK_RETREAT_DUR),
        {"cheek_phase": "retreat"},
    ))

    plan = MotionPlan(say=say, actions=[a for a, _ in tagged])
    plan._action_meta = [m for _, m in tagged]   # type: ignore[attr-defined]
    return plan


def _blush_plan() -> MotionPlan:
    return _build_cheek_plan(
        "Applying blush.",
        [CHEEK_L_CENTER, CHEEK_R_CENTER],
    )


# Registry — add new actions here as the project grows
MAKEUP_ACTIONS: dict[str, dict[str, Any]] = {
    "lipstick": {
        "plan":   _lipstick_plan,
        "region": "lips",
        "joints": {"horizontal": "_1", "vertical": "_2"},
        "label":  "Lipstick",
    },
    "left_eyebrow": {
        "plan":   _left_eyebrow_plan,
        "region": "left_eyebrow",
        "joints": {"horizontal": "_1", "vertical": "_2"},
        "label":  "Left Brow",
    },
    "right_eyebrow": {
        "plan":   _right_eyebrow_plan,
        "region": "right_eyebrow",
        "joints": {"horizontal": "_1", "vertical": "_2"},
        "label":  "Right Brow",
    },
    "both_eyebrows": {
        "plan":   _both_eyebrows_plan,
        "region": "both_eyebrows",
        "joints": {"horizontal": "_1", "vertical": "_2"},
        "label":  "Both Brows",
    },
    "blush": {
        "plan":   _blush_plan,
        "region": "both_cheeks",
        "joints": {"horizontal": "_1", "vertical": "_2"},
        "label":  "Blush",
    },
    # Future actions:
    # "eyeliner": { "plan": _eyeliner_plan, "region": "left_eye", ... },
}


# ---------------------------------------------------------------------------
# Streaming executor — wraps MotionExecutor and emits joint angles over WS
# ---------------------------------------------------------------------------

class StreamingExecutor:
    """Wraps MotionExecutor; intercepts _snap_to to stream angles to browser."""

    def __init__(self, base_executor: MotionExecutor, ws: WebSocketServerProtocol, action: str, color: str):
        self._exec         = base_executor
        self._ws           = ws
        self._action       = action
        self._color        = color
        self._current_meta = {}  # phase metadata for current action — sent with every frame

        original_snap = base_executor._snap_to

        async def _streaming_snap(pose: dict[str, float]) -> None:
            original_snap(pose)
            msg = {
                "type":   "joint_update",
                "action": self._action,
                "color":  self._color,
                "joints": pose,
            }
            msg.update(self._current_meta)  # includes brow_phase in every frame
            try:
                await self._ws.send(json.dumps(msg))
            except Exception:
                pass

        self._async_snap = _streaming_snap
        self._original_snap = original_snap

    async def execute_streaming(self, plan: MotionPlan) -> None:
        """Run the plan step by step, streaming angles after each ramp frame."""
        if plan.say:
            await self._ws.send(json.dumps({"type": "say", "text": plan.say}))

        meta_list = getattr(plan, '_action_meta', [{}] * len(plan.actions))

        for i, action in enumerate(plan.actions, 1):
            meta = meta_list[i - 1] if i - 1 < len(meta_list) else {}
            self._current_meta = meta  # update so all joint_update frames carry this phase
            print(f"  [phase] step {i}: {meta.get('brow_phase', 'none')} action={action.type}")
            msg = {
                "type":        "action_start",
                "step":        i,
                "total":       len(plan.actions),
                "action_type": action.type,
                "color":       self._color,
            }
            msg.update(meta)
            await self._ws.send(json.dumps(msg))
            await self._run_action_streaming(action)

        self._current_meta = {}
        await self._ws.send(json.dumps({"type": "done"}))

    async def _run_action_streaming(self, action: Action) -> None:
        executor = self._exec

        if action.type == "wait":
            await asyncio.sleep(action.duration)
            return

        if action.type == "home":
            await self._ramp_streaming({j: 0.0 for j in executor._current_pose}, action.duration)
            return

        if action.type == "set_joint":
            assert action.joint and action.angle is not None
            from motion import clamp, JOINTS
            target = clamp(action.joint, action.angle, executor.joint_limits)
            new_pose = {**executor._current_pose, action.joint: target}
            await self._ramp_streaming(new_pose, action.duration)
            return

        if action.type == "set_pose":
            assert action.pose
            from motion import clamp
            new_pose = dict(executor._current_pose)
            for j, v in action.pose.items():
                new_pose[j] = clamp(j, v, executor.joint_limits)
            await self._ramp_streaming(new_pose, action.duration)
            return

    async def _ramp_streaming(self, target_pose: dict[str, float], duration: float) -> None:
        executor = self._exec
        if duration <= 0:
            await self._async_snap(target_pose)
            return

        steps = max(2, int(duration * executor.ramp_hz))
        start = dict(executor._current_pose)
        dt    = duration / steps

        for s in range(1, steps + 1):
            t      = s / steps
            interp = {
                j: start.get(j, 0.0) + (target_pose[j] - start.get(j, 0.0)) * t
                for j in target_pose
            }
            await self._async_snap(interp)
            await asyncio.sleep(dt)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

class RobotServer:
    def __init__(self, dry_run: bool = False):
        self.dry_run  = dry_run
        self.executor: MotionExecutor | None = None
        self.cw       = None
        self._lock    = asyncio.Lock()

    def connect_robot(self) -> None:
        if self.dry_run:
            print("  [dry-run] no robot connection")
            return

        from cyberwave import Cyberwave
        print("→ Connecting to Cyberwave…")
        self.cw = Cyberwave()
        self.cw.affect(CW_MODE)
        robot = self.cw.twin(TWIN_ASSET_KEY, twin_id=CW_TWIN_ID, environment_id=CW_ENV_ID)
        self.executor = MotionExecutor(robot)
        print("→ Homing…")
        self.executor.home(duration=1.0)
        time.sleep(0.5)
        print("✅ Robot ready.")

    async def handle(self, ws: WebSocketServerProtocol) -> None:
        print(f"  ◉ client connected: {ws.remote_address}")
        try:
            # Send available actions to browser on connect
            await ws.send(json.dumps({
                "type":    "ready",
                "actions": {k: {"label": v["label"], "region": v["region"]}
                            for k, v in MAKEUP_ACTIONS.items()},
            }))

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "apply":
                    action_key = msg.get("action", "lipstick")
                    color      = msg.get("color", "#cc0000")
                    await self._apply_makeup(ws, action_key, color)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"  ○ client disconnected: {ws.remote_address}")

    async def _apply_makeup(self, ws: WebSocketServerProtocol, action_key: str, color: str) -> None:
        if action_key not in MAKEUP_ACTIONS:
            await ws.send(json.dumps({"type": "error", "message": f"Unknown action: {action_key}"}))
            return

        async with self._lock:  # one motion at a time
            action_def = MAKEUP_ACTIONS[action_key]
            plan       = action_def["plan"]()  # fresh plan each call

            await ws.send(json.dumps({
                "type":   "motion_start",
                "action": action_key,
                "region": action_def["region"],
                "joints": action_def["joints"],
                "color":  color,
            }))

            if self.executor is None:
                # Dry-run: simulate angle stream
                await self._dry_run_stream(ws, plan, action_key, color)
            else:
                streamer = StreamingExecutor(self.executor, ws, action_key, color)
                await streamer.execute_streaming(plan)

    async def _dry_run_stream(
        self, ws: WebSocketServerProtocol, plan: MotionPlan, action: str, color: str
    ) -> None:
        """Simulate joint angle stream without a real robot."""
        import math
        if plan.say:
            await ws.send(json.dumps({"type": "say", "text": plan.say}))

        # Simulate the lipstick stroke angles over time
        total_steps = 60
        for i in range(total_steps + 1):
            t     = i / total_steps
            angle = math.sin(t * math.pi * 2) * 10.0  # sweeps ±10°
            await ws.send(json.dumps({
                "type":   "joint_update",
                "action": action,
                "color":  color,
                "joints": {"_1": angle, "_2": -25.0, "_3": 15.0},
            }))
            await asyncio.sleep(0.05)

        await ws.send(json.dumps({"type": "done"}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if not dry_run and not all([CYBERWAVE_API_KEY, CW_TWIN_ID, CW_ENV_ID]):
        print("❌ Set CYBERWAVE_API_KEY, CYBERWAVE_TWIN_ID, CYBERWAVE_ENVIRONMENT_ID in .env")
        print("   Or run with --dry-run to test without a robot.")
        sys.exit(1)

    server = RobotServer(dry_run=dry_run)
    server.connect_robot()

    print(f"\n🚀 WebSocket server running on ws://{HOST}:{PORT}")
    print(f"   Open index.html in your browser to use the app.\n")

    async with websockets.serve(server.handle, HOST, PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())