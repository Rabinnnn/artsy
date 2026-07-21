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

def _lipstick_plan() -> MotionPlan:
    return MotionPlan(
        say="Applying lipstick.",
        actions=[
            Action(type="set_pose",  pose={"_2": -25.0, "_3": 15.0}, duration=1.2),
            Action(type="wait",      duration=0.4),
            Action(type="set_joint", joint="_1", angle=10.0,  duration=0.6),
            Action(type="set_joint", joint="_1", angle=-10.0, duration=0.6),
            Action(type="set_joint", joint="_1", angle=0.0,   duration=0.6),
            Action(type="home",      duration=1.5),
        ],
    )


# Registry — add new actions here as the project grows
MAKEUP_ACTIONS: dict[str, dict[str, Any]] = {
    "lipstick": {
        "plan":    _lipstick_plan,   # callable so each call gets a fresh plan
        "region":  "lips",
        "joints":  {"horizontal": "_1", "vertical": "_2"},
        "label":   "Lipstick",
    },
    # Future actions — uncomment and implement when ready:
    # "eyeliner": {
    #     "plan":   _eyeliner_plan,
    #     "region": "left_eye",
    #     "joints": {"horizontal": "_1", "vertical": "_2"},
    #     "label":  "Eyeliner",
    # },
    # "blush": {
    #     "plan":   _blush_plan,
    #     "region": "left_cheek",
    #     "joints": {"horizontal": "_1", "vertical": "_2"},
    #     "label":  "Blush",
    # },
}


# ---------------------------------------------------------------------------
# Streaming executor — wraps MotionExecutor and emits joint angles over WS
# ---------------------------------------------------------------------------

class StreamingExecutor:
    """Wraps MotionExecutor; intercepts _snap_to to stream angles to browser."""

    def __init__(self, base_executor: MotionExecutor, ws: WebSocketServerProtocol, action: str, color: str):
        self._exec   = base_executor
        self._ws     = ws
        self._action = action
        self._color  = color

        # Monkey-patch _snap_to to intercept every frame
        original_snap = base_executor._snap_to

        async def _streaming_snap(pose: dict[str, float]) -> None:
            original_snap(pose)
            msg = json.dumps({
                "type":   "joint_update",
                "action": self._action,
                "color":  self._color,
                "joints": pose,
            })
            try:
                await self._ws.send(msg)
            except Exception:
                pass  # client disconnected mid-motion

        # Store async snap; we'll call it from an async runner
        self._async_snap = _streaming_snap
        self._original_snap = original_snap

    async def execute_streaming(self, plan: MotionPlan) -> None:
        """Run the plan step by step, streaming angles after each ramp frame."""
        if plan.say:
            await self._ws.send(json.dumps({"type": "say", "text": plan.say}))

        for i, action in enumerate(plan.actions, 1):
            await self._ws.send(json.dumps({
                "type": "action_start",
                "step": i,
                "total": len(plan.actions),
                "action_type": action.type,
            }))
            await self._run_action_streaming(action)

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