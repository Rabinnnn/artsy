"""Voice/Text-driven SO-101 makeup controller.

Speak or type a request. DeepSeek classifies it against a fixed menu of
makeup actions — lipstick, left/right/both eyebrows, blush — and, if it
matches, dispatches that exact choreographed motion (with an optional
colour, e.g. "put some red lipstick on") to server.py, which runs it on the
real (or dry-run) arm and streams it to any connected browser for
visualization. Requests that don't match one of these actions are politely
declined — this controller only does makeup, nothing else.

This script never talks to Cyberwave directly — server.py owns the one
robot connection. This script is just another WebSocket client of it,
exactly like the browser is, so whatever it triggers shows up live in the
web app too.

    python server.py                                  # start this first
    python nl_arm_controller.py                       # text REPL
    python nl_arm_controller.py --voice               # voice REPL (hold SPACE)
    python nl_arm_controller.py --dry-run             # classify only, don't connect to server.py
    python nl_arm_controller.py --check               # env + deps self-check

Examples to say or type:
    put some red lipstick on
    put some pink lipstick on
    draw both my eyebrows
    fill in just my left brow with dark brown
    add a soft pink blush

`exit`, `quit`, `bye`, or `Ctrl+C` to leave.
In voice mode, Esc cancels the *current* recording without exiting.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Config (from env)
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
MISTRAL_STT_MODEL = os.environ.get("MISTRAL_STT_MODEL", "voxtral-mini-latest")

VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "false").lower() == "true"
SAMPLE_RATE = 16000

# server.py's own WS listener — see HOST/PORT constants in server.py
ROBOT_SERVER_WS_URL = os.environ.get("ROBOT_SERVER_WS_URL", "ws://localhost:8765")

EXIT_WORDS = {"exit", "quit", "bye", "stop the demo", "shutdown"}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _check_secret(name: str, value: str | None) -> tuple[str, bool]:
    if not value:
        return f"  {name:<24} ❌ not set", False
    return f"  {name:<24} ✅ {value[:8]}…  (len {len(value)})", True


def run_self_check() -> int:
    print("─" * 64)
    print("  NL → SO-101 Makeup Controller — environment self-check")
    print("─" * 64)

    rows = [
        _check_secret("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
        _check_secret("MISTRAL_API_KEY", MISTRAL_API_KEY),
    ]
    for line, _ in rows:
        print(line)
    keys_ok = all(ok for _, ok in rows)

    print()
    print(f"  ROBOT_SERVER_WS_URL      = {ROBOT_SERVER_WS_URL}")
    print(f"  DEEPSEEK_MODEL           = {DEEPSEEK_MODEL}")
    print(f"  MISTRAL_STT_MODEL        = {MISTRAL_STT_MODEL}")
    print(f"  VOICE_ENABLED            = {VOICE_ENABLED}")

    print()
    deps_ok = True
    for mod_name in ("websockets", "deepseek", "httpx", "sounddevice", "soundfile", "pynput"):
        try:
            __import__(mod_name)
            print(f"  import {mod_name:<14} ✅")
        except ImportError as exc:
            print(f"  import {mod_name:<14} ❌  ({exc})")
            deps_ok = False

    print("─" * 64)
    if keys_ok and deps_ok:
        print("  ✅ Environment ready. (Make sure server.py is running separately.)")
        return 0
    print("  ❌ Fix the items marked ❌ above and re-run.")
    return 1


# ---------------------------------------------------------------------------
# WebSocket client — mirrors what the browser does, but driven by voice/text
# ---------------------------------------------------------------------------


class RobotConnection:
    """Thin WebSocket client for server.py. Sends `apply` messages exactly
    like the browser's Apply button does, and prints the resulting status
    stream — the actual drawing happens in the browser, which is watching
    the same broadcast.
    """

    def __init__(self, ws) -> None:
        self._ws = ws
        self.available_actions: dict[str, dict] = {}

    async def read_ready(self) -> None:
        raw = await self._ws.recv()
        msg = json.loads(raw)
        if msg.get("type") == "ready":
            self.available_actions = msg.get("actions", {})

    async def apply_makeup(self, action_key: str, color: str) -> bool:
        """Send the action and print status until it completes. Returns
        True on success, False on error or connection loss — never raises,
        so one bad response can't take down the whole REPL.
        """
        import websockets

        try:
            await self._ws.send(json.dumps({
                "type": "apply", "action": action_key, "color": color,
            }))
            return await self._drain_until_done()
        except websockets.exceptions.ConnectionClosed as exc:
            print(f"  ❌ lost connection to server.py: code={exc.code} reason={exc.reason!r}")
            print("     Check server.py's own terminal for what it logged at this moment.")
            return False
        except Exception as exc:
            print(f"  ❌ lost connection to server.py: {exc!r}")
            return False

    async def _drain_until_done(self) -> bool:
        while True:
            raw = await self._ws.recv()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "say":
                print(f"  💬  {msg.get('text', '')}")
            elif mtype == "action_start":
                phase = msg.get("brow_phase") or msg.get("cheek_phase") or msg.get("action_type")
                print(f"  ▶  step {msg.get('step')}/{msg.get('total')}  ({phase})")
            elif mtype == "error":
                print(f"  ❌  {msg.get('message', 'unknown error')}")
                return False
            elif mtype == "done":
                print("  ✅  done")
                return True
            # joint_update / motion_start: nothing to print — the browser draws from these


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _print_banner(dry_run: bool, voice: bool, actions: dict) -> None:
    print("─" * 64)
    print(f"  NL → SO-101 Makeup Controller  ({'voice' if voice else 'text'})")
    print("─" * 64)
    print(f"  mode:        {'DRY-RUN (no server connection)' if dry_run else ROBOT_SERVER_WS_URL}")
    print(f"  planner:     {DEEPSEEK_MODEL}")
    if voice:
        print(f"  STT model:   {MISTRAL_STT_MODEL}")
    print()
    print("  This controller only applies makeup — nothing else:")
    for key, info in (actions or {}).items():
        print(f"    • {info.get('label', key)}")
    print()
    print("  Examples:")
    print("    • put some red lipstick on")
    print("    • put some pink lipstick on")
    print("    • draw both my eyebrows")
    print("    • add a soft pink blush")
    if voice:
        print("  Hold SPACE while speaking, release to send. Esc cancels a turn.")
    print(f"  Exit: {'say' if voice else 'type'} {sorted(EXIT_WORDS)} or press Ctrl+C.")
    if not dry_run:
        print("  Open index.html in a browser (connected to the same server.py) to watch it live.")
    print("─" * 64)


def _read_text() -> str | None:
    try:
        return input("\n  you ▸ ").strip()
    except EOFError:
        print()
        return None


def _read_voice() -> str | None:
    from voice import capture_utterance

    transcript, err = capture_utterance()
    if err:
        return ""  # keep the loop alive; user retries by holding SPACE again
    return transcript


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def run_agent(dry_run: bool, voice: bool) -> int:
    if not DEEPSEEK_API_KEY:
        print("❌ DEEPSEEK_API_KEY not set in .env")
        return 1

    if voice and not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not set in .env (required for --voice)")
        return 1

    from planner import classify_makeup_intent

    conn: RobotConnection | None = None
    ws_ctx = None

    if not dry_run:
        import websockets

        print(f"→ Connecting to {ROBOT_SERVER_WS_URL}…")
        try:
            ws_ctx = websockets.connect(ROBOT_SERVER_WS_URL)
            ws = await ws_ctx.__aenter__()
        except Exception as exc:
            print(f"  ❌ Could not connect to server.py: {exc}")
            print("     Make sure `python server.py` (or `--dry-run`) is running.")
            return 1
        conn = RobotConnection(ws)
        await conn.read_ready()
        print(f"  ✓ connected — {len(conn.available_actions)} makeup actions available")

    _print_banner(dry_run, voice, conn.available_actions if conn else {})

    try:
        while True:
            # input()/voice capture are blocking — run them off the event
            # loop so this connection keeps answering the server's WebSocket
            # ping/pong keepalive while we're waiting for the user, instead
            # of going quiet and looking like a dead connection.
            utterance = await asyncio.to_thread(_read_voice if voice else _read_text)
            if utterance is None:
                break

            if not utterance:
                continue
            if utterance.lower().rstrip(".!?") in EXIT_WORDS:
                break

            # Everything below is wrapped so a single bad utterance/response
            # can never crash the whole REPL — worst case, we print an error
            # and loop back for the next command.
            try:
                t0 = time.monotonic()
                # classify_makeup_intent() makes a synchronous DeepSeek API
                # call that can take anywhere from ~2s to 30+ seconds. Same
                # reasoning as above: run it off the event loop so we don't
                # go silent on the WebSocket connection while waiting for it
                # — that silence was killing the connection (code 1006)
                # before the resulting 'apply' message ever got sent.
                intent = await asyncio.to_thread(classify_makeup_intent, utterance)
                dt = (time.monotonic() - t0) * 1000

                if not intent.is_makeup or not intent.action:
                    print(f"  💬  {intent.say}  ({dt:.0f} ms)")
                    if intent.error:
                        print(f"     (classifier note: {intent.error})")
                    continue

                print(f"  🎨  {intent.action}  (color {intent.color}, {dt:.0f} ms)")
                print(f"  💬  {intent.say}")

                if dry_run:
                    continue

                await conn.apply_makeup(intent.action, intent.color)

            except Exception as exc:
                print(f"  ❌ unexpected error handling that command: {exc}")
                continue
    except KeyboardInterrupt:
        print("\n  (Ctrl+C — shutting down)")
    finally:
        if ws_ctx is not None:
            try:
                await ws_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    print("👋 bye")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(run_self_check())

    dry_run = "--dry-run" in sys.argv
    voice = "--voice" in sys.argv
    sys.exit(asyncio.run(run_agent(dry_run=dry_run, voice=voice)))


if __name__ == "__main__":
    main()