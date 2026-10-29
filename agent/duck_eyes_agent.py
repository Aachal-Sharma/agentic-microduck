"""
Microduck with eyes: see -> think -> act.

Every step, the duck's head camera image is sent to a vision-language model,
which says whether the target is visible and where it is in the image. A small
controller turns that into turn / walk / steer, and the trained RL policy in
infer_policy.py does the actual walking. Then it looks again.

Run from inside the microduck_rl folder:
    export NVIDIA_API_KEY=your_key
    uv run --with openai python duck_eyes_agent.py
"""

import base64
import json
import os
import pty
import subprocess
import sys
import threading
import time

from openai import OpenAI

SCENE = "src/mjlab_microduck/robot/microduck/scene_eyes.xml"
POLICY = "pretrained/velstand.onnx"
VIEW_FILE = "duck_view.png"
STATUS_FILE = "duck_status.json"  # read by duck_dashboard.py
# any vision model on build.nvidia.com works; set DUCK_MODEL to try another one
MODEL = os.environ.get("DUCK_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
MAX_STEPS = 40

UP = "\x1b[A"
STOP = " "

# timings, in seconds
FORWARD_TIME = 1.5
SEARCH_TURN_TIME = 0.6   # used when the target isn't visible
SETTLE_TIME = 0.8        # let it stop wobbling before taking the next picture

# Turning model, measured roughly from the first runs: the policy needs about
# a quarter second before it starts rotating, then turns at roughly this rate.
TURN_LAG = 0.25          # seconds before the turn really starts
TURN_RATE = 130.0        # degrees per second once it's turning
CAMERA_HFOV = 58.0       # horizontal field of view of the eye camera, degrees

# where the target is across the image: 0 = left edge, 1 = right edge
CENTER_BAND = 0.12       # within this of the middle -> just walk straight
STEER_BAND = 0.25        # within this -> walk and steer at the same time

PROMPT = """You are the eyes of a small robot duck. This image is from the
camera in its head, looking forward.

Target: {goal}

Answer these about the target only:
- visible: true or false
- x: horizontal position of the target's centre in the image, from 0.0
  (left edge) to 1.0 (right edge). Use 0.5 if it is in the middle.
- close: true only if the target looks big, taking up a large part of the image

Reply with only JSON, like:
{{"see": "<short description>", "visible": true, "x": 0.62, "close": false}}"""


class Duck:
    """Starts the simulator in a pseudo-terminal and sends it keys."""

    def __init__(self):
        cmd = [
            sys.executable, "duck_eyes_sim.py",
            "--walking", POLICY,
            "--new-cmd-obs",
            "--scene", SCENE,
        ]
        env = dict(os.environ)
        env.setdefault("MUJOCO_GL", "egl")  # offscreen camera rendering

        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, env=env)
        os.close(slave)

        self.log = open("duck_sim.log", "wb")
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while True:
            try:
                data = os.read(self.master, 1024)
            except OSError:
                break
            if not data:
                break
            self.log.write(data)
            self.log.flush()

    def send(self, keys):
        os.write(self.master, keys.encode())

    def turn(self, direction, seconds):
        self.send(STOP + ("a" if direction == "left" else "e"))
        time.sleep(seconds)
        self.send(STOP)
        time.sleep(SETTLE_TIME)

    def forward(self, steer=None, steer_time=0.0):
        self.send(STOP + UP)
        if steer:
            # steer only for a moment, then keep walking straight
            self.send("a" if steer == "left" else "e")
            time.sleep(steer_time)
            self.send(STOP + UP)
            time.sleep(max(FORWARD_TIME - steer_time, 0.3))
        else:
            time.sleep(FORWARD_TIME)
        self.send(STOP)
        time.sleep(SETTLE_TIME)

    def close(self):
        self.send("q")
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def look(client, goal):
    with open(VIEW_FILE, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    prompt = PROMPT.format(goal=goal)
    image_url = f"data:image/png;base64,{image_b64}"

    if "llama" in MODEL:
        # NVIDIA's Llama vision endpoint takes the image as an <img> tag
        content = prompt + f' <img src="{image_url}" />'
        extra = {}
    else:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        extra = {}
        if "qwen" in MODEL:
            # skip long step-by-step thinking, we only need one quick decision
            extra = {"chat_template_kwargs": {"thinking": False}}

    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=800,
        extra_body=extra,
    )
    text = reply.choices[0].message.content or ""
    return parse_decision(text)


def parse_decision(text):
    """Pull the JSON out of the model's reply. Returns None if it can't."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        seen = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if "visible" not in seen:
        return None
    return seen


def turn_time_for(offset):
    """How long to hold the turn key to centre a target that is `offset`
    away from the middle of the image (offset in -0.5 .. 0.5)."""
    degrees = abs(offset) * CAMERA_HFOV
    return TURN_LAG + degrees / TURN_RATE


def decide(seen):
    """Turn what the duck sees into a move. Returns (label, move), where move
    is a function that makes the duck do it (None when the goal is reached)."""
    if not seen.get("visible"):
        return "searching (turn left)", lambda duck: duck.turn("left", SEARCH_TURN_TIME)

    try:
        offset = float(seen.get("x", 0.5)) - 0.5
    except (TypeError, ValueError):
        offset = 0.0

    if seen.get("close") and abs(offset) < STEER_BAND:
        return "done", None

    side = "left" if offset < 0 else "right"
    if abs(offset) < CENTER_BAND:
        return "forward", lambda duck: duck.forward()

    t = turn_time_for(offset)
    if abs(offset) < STEER_BAND:
        return (f"forward + steer {side} ({t:.2f} s)",
                lambda duck: duck.forward(steer=side, steer_time=t))
    return f"turn {side} ({t:.2f} s)", lambda duck: duck.turn(side, t)


class Status:
    """Keeps the dashboard up to date with what the agent is doing."""

    def __init__(self):
        self.state = {"model": MODEL, "goal": "-", "stage": "idle",
                      "see": "waiting for a mission", "action": "-", "history": []}
        self.save()

    def update(self, **changes):
        self.state.update(changes)
        self.save()

    def log(self, line):
        self.state["history"] = (self.state["history"] + [line])[-8:]
        self.save()

    def save(self):
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, STATUS_FILE)


def wait_for_camera(timeout=30):
    start = time.time()
    while not os.path.exists(VIEW_FILE):
        if time.time() - start > timeout:
            sys.exit("No camera image yet. Check duck_sim.log for errors.")
        time.sleep(0.5)


def main():
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("Set NVIDIA_API_KEY first.")

    if os.path.exists(VIEW_FILE):
        os.remove(VIEW_FILE)

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key, timeout=40)
    duck = Duck()
    wait_for_camera()
    status = Status()
    dashboard = subprocess.Popen([sys.executable, "duck_dashboard.py"])
    time.sleep(5)  # let the duck settle after spawning

    try:
        while True:
            status.update(stage="idle", see="waiting for a mission", action="-")
            goal = input("\nWhat should the duck find? (or 'quit')\n> ").strip()
            if not goal:
                continue
            if goal.lower() in ("quit", "exit", "q"):
                break

            status.update(goal=goal, history=[], visible=False, x=None)
            status.log(f"new mission: find {goal}")

            for step in range(1, MAX_STEPS + 1):
                status.update(step=step, stage="perceive", action="capturing camera view")
                time.sleep(0.3)  # long enough to see the stage change on the dashboard

                status.update(stage="reason", action=f"asking {MODEL.split('/')[-1]}")
                decision = None
                for attempt in range(2):  # one retry if the reply is unusable
                    try:
                        decision = look(client, goal)
                    except Exception as e:
                        print(f"step {step}: vision model error: {e}")
                    if decision:
                        break

                if not decision:
                    print(f"step {step}: no usable answer, looking again")
                    status.log(f"{step}. no usable answer, retrying")
                    continue

                see = decision.get("see", "?")
                x = decision.get("x")
                visible = bool(decision.get("visible"))
                where = f" (x={x:.2f})" if visible and isinstance(x, (int, float)) else ""

                label, move = decide(decision)
                print(f"step {step}: {see}{where} -> {label}")
                status.update(stage="act", see=see, visible=visible, x=x, action=label)
                status.log(f"{step}. {label}{where}")

                if move is None:
                    status.update(stage="done", action="target reached")
                    status.log("mission complete")
                    print("reached the target")
                    time.sleep(3)
                    break
                move(duck)
            else:
                print("gave up after max steps")
                status.log("gave up after max steps")
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        duck.close()
        dashboard.terminate()
        dashboard.wait(timeout=5)


if __name__ == "__main__":
    main()
