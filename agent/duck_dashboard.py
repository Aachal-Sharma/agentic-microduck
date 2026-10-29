"""
Dashboard for the Microduck vision agent, in one window:
  - left:  the simulation, from a camera that follows the duck
  - right: what the duck sees, and the agent's perceive -> reason -> act loop

Press R to start/stop recording to an MP4 in the recordings/ folder.
Press Q (or close the window) to close the dashboard.

duck_eyes_agent.py starts this for you. It reads files that the simulator
and the agent keep updating:
    duck_scene.png    - outside view of the duck
    duck_view.png     - the duck's eye camera
    duck_status.json  - goal, current stage, what the model saw, action, log
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCENE_FILE = "duck_scene.png"
VIEW_FILE = "duck_view.png"
STATUS_FILE = "duck_status.json"
RECORD_DIR = "recordings"
WINDOW = "Agentic Microduck"

W, H = 1280, 720
FPS = 15

# colours (RGB)
BG = (18, 22, 30)
CARD = (30, 36, 48)
LINE = (60, 70, 90)
TEXT = (230, 234, 240)
MUTED = (150, 160, 178)
ACCENT = (255, 196, 60)      # duck yellow
GREEN = (90, 210, 120)
RED = (235, 70, 70)
STAGE_COLOURS = {
    "perceive": (90, 170, 255),
    "reason": (190, 130, 255),
    "act": (255, 196, 60),
}


def load_font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"]
    folders = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"]
    for folder in folders:
        for name in names:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = load_font(24, bold=True)
F_SUB = load_font(15)
F_LABEL = load_font(13, bold=True)
F_BODY = load_font(16)
F_BODY_B = load_font(17, bold=True)
F_SMALL = load_font(13)
F_CHIP = load_font(15, bold=True)


def read_image(path, size):
    img = cv2.imread(path) if os.path.exists(path) else None
    if img is None:
        return Image.new("RGB", size, CARD)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img


def read_status():
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def wrap(draw, text, font, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) > width and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    return lines


def card(draw, box, label=None):
    draw.rounded_rectangle(box, radius=12, fill=CARD, outline=LINE)
    if label:
        draw.text((box[0] + 14, box[1] + 10), label.upper(), font=F_LABEL, fill=MUTED)


def draw_frame(status, recording, rec_start):
    frame = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(frame)

    # header
    d.text((24, 14), "Agentic Microduck", font=F_TITLE, fill=TEXT)
    title_w = d.textlength("Agentic Microduck", font=F_TITLE)
    d.text((24 + title_w + 18, 21), "vision-language agent  +  RL locomotion policy",
           font=F_SUB, fill=MUTED)

    right = W - 24
    model = status.get("model", "")
    if model:
        tw = d.textlength(model, font=F_SMALL)
        d.text((right - tw, 22), model, font=F_SMALL, fill=MUTED)
        right -= tw + 28
    if recording:
        secs = int(time.time() - rec_start)
        rec = f"REC {secs // 60:02d}:{secs % 60:02d}"
        rw = d.textlength(rec, font=F_LABEL)
        d.text((right - rw, 20), rec, font=F_LABEL, fill=RED)
        d.ellipse((right - rw - 22, 21, right - rw - 8, 35), fill=RED)

    # left: simulation view
    card(d, (16, 60, 776, 652), "MuJoCo simulation")
    frame.paste(read_image(SCENE_FILE, (736, 552)), (28, 88))

    # bottom left: mission bar
    card(d, (16, 662, 776, 708))
    goal = status.get("goal", "-")
    stage = status.get("stage", "idle")
    if stage == "done":
        state, colour = "Target reached", GREEN
    elif stage == "idle":
        state, colour = "Waiting for a mission", MUTED
    elif not status.get("visible"):
        state, colour = "Searching", ACCENT
    else:
        state, colour = "Approaching target", ACCENT
    d.text((32, 674), "Mission:", font=F_BODY_B, fill=MUTED)
    d.text((118, 674), f"find {goal}" if goal != "-" else "-", font=F_BODY_B, fill=TEXT)
    sw = d.textlength(state, font=F_BODY_B)
    d.text((760 - sw, 674), state, font=F_BODY_B, fill=colour)

    # right top: duck's eye view
    card(d, (792, 60, 1264, 400), "Duck's eye view  (agent input)")
    eye = read_image(VIEW_FILE, (400, 300))
    ed = ImageDraw.Draw(eye)
    if status.get("visible") and isinstance(status.get("x"), (int, float)):
        x = int(min(max(status["x"], 0.0), 1.0) * 400)
        ed.line((x, 0, x, 300), fill=ACCENT, width=2)
        ed.ellipse((x - 14, 136, x + 14, 164), outline=ACCENT, width=3)
    ed.line((200, 284, 200, 300), fill=TEXT, width=1)
    frame.paste(eye, (828, 88))

    # right bottom: agent loop
    card(d, (792, 410, 1264, 708), "Agent loop")
    chips = [("perceive", "PERCEIVE"), ("reason", "REASON"), ("act", "ACT")]
    cx = 808
    for i, (key, name) in enumerate(chips):
        active = stage == key
        cw = 118
        box = (cx, 440, cx + cw, 474)
        if active:
            d.rounded_rectangle(box, radius=17, fill=STAGE_COLOURS[key])
            d.text((cx + (cw - d.textlength(name, font=F_CHIP)) / 2, 448), name, font=F_CHIP, fill=BG)
        else:
            d.rounded_rectangle(box, radius=17, outline=LINE, width=2)
            d.text((cx + (cw - d.textlength(name, font=F_CHIP)) / 2, 448), name, font=F_CHIP, fill=MUTED)
        if i < 2:
            d.text((cx + cw + 10, 446), "→", font=F_BODY_B, fill=MUTED)
        cx += cw + 36
    d.text((cx - 22, 446), "↺", font=F_BODY_B, fill=MUTED)

    y = 490
    d.text((808, y), "Thought", font=F_LABEL, fill=MUTED)
    for line in wrap(d, status.get("see", "-"), F_BODY, 440)[:2]:
        y += 20
        d.text((808, y), line, font=F_BODY, fill=TEXT)

    y += 30
    d.text((808, y), "Action", font=F_LABEL, fill=MUTED)
    action = status.get("action", "-")
    d.text((808, y + 18), action, font=F_BODY_B, fill=GREEN if stage == "done" else ACCENT)

    y += 52
    d.line((808, y, 1248, y), fill=LINE)
    for entry in status.get("history", [])[-4:]:
        y += 6
        text = entry if d.textlength(entry, font=F_SMALL) < 440 else entry[:62] + "…"
        d.text((808, y), text, font=F_SMALL, fill=MUTED)
        y += 16

    return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)


class Recorder:
    """Records dashboard frames to MP4. Uses ffmpeg (H.264, high quality) when
    it's installed, and falls back to OpenCV's own encoder otherwise."""

    def __init__(self):
        self.proc = None
        self.writer = None
        self.path = None
        self.start = 0.0

    @property
    def on(self):
        return self.proc is not None or self.writer is not None

    def toggle(self):
        if self.on:
            self.stop()
            return
        os.makedirs(RECORD_DIR, exist_ok=True)
        self.path = os.path.join(RECORD_DIR, datetime.now().strftime("duck_agent_%Y%m%d_%H%M%S.mp4"))

        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                   "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   self.path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("ffmpeg not found, recording with OpenCV (lower quality)")
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        self.start = time.time()
        print(f"recording to {self.path}")

    def write(self, frame):
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        elif self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
            print(f"saved {self.path}")
        elif self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"saved {self.path}")


def main():
    # when the agent closes us, still finish writing any recording
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    rec = Recorder()
    try:
        while True:
            t0 = time.time()
            frame = draw_frame(read_status(), rec.on, rec.start)
            cv2.imshow(WINDOW, frame)
            rec.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                rec.toggle()
            elif key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            # keep a steady frame rate so the recording plays at normal speed
            time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))
    finally:
        rec.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
