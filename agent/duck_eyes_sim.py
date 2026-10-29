"""
Runs scripts/infer_policy.py unchanged, but also saves what the duck's head
camera sees to duck_view.png a few times per second.

It also saves an outside view that follows the duck to duck_scene.png,
which the dashboard shows and records.

It works by wrapping mujoco.viewer.launch_passive, so every time the viewer
syncs we also render these two views offscreen.

Run from inside the microduck_rl folder (duck_eyes_agent.py starts it for you):
    uv run python duck_eyes_sim.py --walking pretrained/velstand.onnx --new-cmd-obs \
        --scene src/mjlab_microduck/robot/microduck/scene_eyes.xml
"""

import math
import os
import runpy
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

VIEW_FILE = "duck_view.png"     # what the duck sees
SCENE_FILE = "duck_scene.png"   # outside view following the duck, for the dashboard
SAVE_EVERY = 1 / 15             # seconds, about 15 frames per second
WIDTH, HEIGHT = 400, 300               # eye view, same size the dashboard shows it
SCENE_WIDTH, SCENE_HEIGHT = 736, 552   # outside view, same size as on the dashboard
EYE_HEIGHT = 0.2   # metres, a bit above the head
EYE_PITCH = -12    # degrees, looking slightly down

_launch_passive = mujoco.viewer.launch_passive


class ViewerWithEyes:
    def __init__(self, viewer, model, data):
        self._viewer = viewer
        self._data = data
        self._trunk = model.body("trunk_base").id
        self._head = model.body("jaw_soft").id
        # the default offscreen buffer is 640x480; make room for the bigger
        # views and use more anti-aliasing samples so edges stay sharp
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 960)
        model.vis.quality.offsamples = 8

        self._renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self._scene_renderer = mujoco.Renderer(model, SCENE_HEIGHT, SCENE_WIDTH)
        self._last_save = 0.0

    def __enter__(self):
        self._viewer.__enter__()
        return self

    def __exit__(self, *exc):
        return self._viewer.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def _eye_camera(self):
        # The built-in head_camera sits behind the beak and mostly sees the
        # duck's own head, so we place a virtual camera just in front of the
        # head, facing the way the body is facing, tilted slightly down.
        trunk_x = self._data.xmat[self._trunk].reshape(3, 3)[:, 0]
        yaw = math.degrees(math.atan2(trunk_x[1], trunk_x[0]))

        forward = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
        eye = self._data.xpos[self._head] + forward * 0.05
        eye[2] = EYE_HEIGHT

        pitch = math.radians(EYE_PITCH)
        view_dir = np.array([math.cos(pitch) * forward[0],
                             math.cos(pitch) * forward[1],
                             math.sin(pitch)])

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.0
        cam.lookat[:] = eye + view_dir * cam.distance
        cam.azimuth = yaw
        cam.elevation = EYE_PITCH
        return cam

    def _follow_camera(self):
        # fixed wide angle from behind, following the duck as it moves
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self._data.xpos[self._trunk]
        cam.lookat[2] = 0.1
        cam.distance = 2.8
        cam.azimuth = 15
        cam.elevation = -40
        return cam

    def sync(self, *args, **kwargs):
        self._viewer.sync(*args, **kwargs)

        now = time.time()
        if now - self._last_save < SAVE_EVERY:
            return
        self._last_save = now

        self._renderer.update_scene(self._data, camera=self._eye_camera())
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        save_image(self._renderer.render(), VIEW_FILE)

        self._scene_renderer.update_scene(self._data, camera=self._follow_camera())
        self._scene_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        save_image(self._scene_renderer.render(), SCENE_FILE)


def save_image(pixels, path):
    tmp = path + ".tmp.png"
    Image.fromarray(pixels).save(tmp, compress_level=1)
    os.replace(tmp, path)  # so readers never get a half-written file


def launch_passive(model, data, *args, **kwargs):
    viewer = _launch_passive(model, data, *args, **kwargs)
    return ViewerWithEyes(viewer, model, data)


mujoco.viewer.launch_passive = launch_passive

sys.argv = ["scripts/infer_policy.py"] + sys.argv[1:]
runpy.run_path("scripts/infer_policy.py", run_name="__main__")
