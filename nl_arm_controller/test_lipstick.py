"""test_lipstick.py — Hardcoded lipstick-application mime test.

Runs the arm through a lipstick motion WITHOUT calling any LLM,
so you can tune the angles freely without burning API tokens.

Usage:
    python test_lipstick.py            # drives the real twin
    python test_lipstick.py --dry-run  # prints the plan, no robot motion

Tweak the angles at the top of build_lipstick_plan() until the motion
looks natural, then we'll wire it into the planner as a few-shot example.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(override=False)

from motion import Action, MotionExecutor, MotionPlan  # noqa: E402


# ---------------------------------------------------------------------------
# Tune these angles until the motion looks right on your arm
# ---------------------------------------------------------------------------

LIFT_SHOULDER   = -25.0   # _2: raise arm toward "face height" (negative = up on SO-101)
EXTEND_ELBOW    =  15.0   # _3: push arm slightly forward
STROKE_LEFT     =  10.0   # _1: small left stroke across "lips"
STROKE_RIGHT    = -10.0   # _1: small right stroke back
LIFT_DURATION   =  1.2    # seconds to raise the arm
STROKE_DURATION =  0.6    # seconds per lip stroke
HOME_DURATION   =  1.5    # seconds to return home


def build_lipstick_plan() -> MotionPlan:
    """Construct the lipstick mime as a hand-crafted MotionPlan."""
    return MotionPlan(
        say="Applying lipstick.",
        actions=[
            # 1. Raise arm to face height + slight forward extension
            Action(
                type="set_pose",
                pose={"_2": LIFT_SHOULDER, "_3": EXTEND_ELBOW},
                duration=LIFT_DURATION,
            ),
            # 2. Pause — "touching" the lips
            Action(type="wait", duration=0.4),
            # 3. Stroke left across lips
            Action(type="set_joint", joint="_1", angle=STROKE_LEFT, duration=STROKE_DURATION),
            # 4. Stroke right across lips
            Action(type="set_joint", joint="_1", angle=STROKE_RIGHT, duration=STROKE_DURATION),
            # 5. Centre again
            Action(type="set_joint", joint="_1", angle=0.0, duration=STROKE_DURATION),
            # 6. Return home
            Action(type="home", duration=HOME_DURATION),
        ],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    dry_run = "--dry-run" in sys.argv

    plan = build_lipstick_plan()

    if dry_run:
        print("=== DRY-RUN: no robot motion ===")
        print(f"say: {plan.say}")
        for i, a in enumerate(plan.actions, 1):
            print(f"  {i}. {a}")
        return

    CYBERWAVE_API_KEY = os.environ.get("CYBERWAVE_API_KEY")
    CW_MODE           = os.environ.get("CW_MODE", "simulation")
    CW_TWIN_ID        = os.environ.get("CYBERWAVE_TWIN_ID")
    CW_ENV_ID         = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
    TWIN_ASSET_KEY    = "the-robot-studio/so101"

    if not all([CYBERWAVE_API_KEY, CW_TWIN_ID, CW_ENV_ID]):
        print("❌ Set CYBERWAVE_API_KEY, CYBERWAVE_TWIN_ID, CYBERWAVE_ENVIRONMENT_ID in .env")
        sys.exit(1)

    from cyberwave import Cyberwave

    print("→ Connecting to Cyberwave…")
    cw = Cyberwave()
    cw.affect(CW_MODE)
    robot = cw.twin(TWIN_ASSET_KEY, twin_id=CW_TWIN_ID, environment_id=CW_ENV_ID)

    executor = MotionExecutor(robot)

    print("→ Homing…")
    executor.home(duration=1.0)
    time.sleep(0.5)

    print("→ Running lipstick plan…")
    executor.execute(plan)
    time.sleep(0.5)

    print("→ Done. Disconnecting…")
    cw.disconnect()
    print("✅ Test complete.")


if __name__ == "__main__":
    main()