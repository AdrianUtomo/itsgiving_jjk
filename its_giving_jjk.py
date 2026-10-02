#!/usr/bin/env python3
"""
its_giving_jjk.py — Jujutsu Kaisen hand signs on top of your face, live in Zoom / Meet.

Same pipeline as gazijarin's its_giving, trimmed to hand signs: MediaPipe tracks your
hands (and your face, to know where to paste); when a sign matches, the matching
image or GIF is pasted over your face and the frame goes out through a virtual
camera. See README.md.

  python its_giving_jjk.py [--camera 1] [--no-vcam] [--size 640x480] [--no-flip] [--max-size 0.6]

Keys:  q quit   d toggle HUD   h toggle hand skeleton   s save a snapshot (preview + hand landmarks) to snapshots/   1-7 force-show a pose

Poses:
  gojo_satoru     Gojo's Unlimited Void: index + middle up and crossed, ring + pinky folded
  ryomen_sukuna   Sukuna's Malevolent Shrine (Enmaten seal), two hands: middle + ring up and meeting the
                  other hand's at a peak, index + pinky curled. Checked as: upright fingertips touching, pinkies curled short
  mahito          Mahito's Self-Embodiment of Perfection, two hands in a diamond: index tips pressed together on top,
                  pinky tips pressed together below, middle + ring interlocked, space between the palms.
                  Checked as: upright fingertips touching, pinkies pointing down with their tips touching
  megumi_mahoraga Megumi summoning Mahoraga: two fists, overlapping, arms crossed.
                    Checked as: both hands fists, close together, each on the other's side (from MediaPipe's handedness)
  hakari_kinji    Hakari's Idle Death Gamble: right hand an OK sign (thumb + index tips in a circle, other fingers
                  raised), left hand open, palm up, near the waist. Checked as: one hand an OK sign, the other open
                  and flat (not raised), lower down. Either hand may do either part, and palm up isn't checked.
  okkotsu_yuta    Yuta's cursed energy beam: right hand a finger gun (index + thumb up, other three curled), left
                  hand just visible. Checked as: one hand a finger gun, a second hand in frame (any shape).
  higuruma_hiromi both hands covering the mouth: both palm centres within
                  0.6 face widths of the mouth.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

POSES = ["gojo_satoru", "ryomen_sukuna", "mahito", "megumi_mahoraga", "hakari_kinji", "okkotsu_yuta", "higuruma_hiromi"]
TEST_KEYS = "1234567"

FACE_SCALE = 2.0
MAX_SIZE = 0.6  # cap on the meme's longer side, as a fraction of the frame height, so it stays off your hands
                # (a 16:9 GIF at 0.6 still covers the face but ends about at the chin; uncapped it hung ~80px below)
HOLD_FRAMES = 10
ARM = {"gojo_satoru": 5, "ryomen_sukuna": 5, "mahito": 5, "megumi_mahoraga": 5, "hakari_kinji": 5, "okkotsu_yuta": 5,
       "higuruma_hiromi": 5}
T = dict(
    cross=0.25,
    press=0.4,
    pinky=1.55,
    pinky_drop=0.15,
    fist=1.0,
    fist_reach=1.3,
    fist_gap=1.8,  # palm centres, in hand sizes; real snapshots of the sign: 1.39-1.50 (fists side by side)
    pinch=0.3,      # thumb tip <-> index tip, in hand sizes
    straight=1.15,  # fold above this counts as a straight finger (real snapshots: straight 1.29-1.47, curled < 1.0)
    tilt=-0.7,      # open hand's wrist->middle knuckle, vertical part: -1 fingers up, 0 sideways, +1 down
    gun_curl=1.1,   # finger gun: middle / ring / pinky fold must all be under this (real snapshots: 0.80-1.01)
    thumb_rise=0.6,  # thumb tip above its base joint, in hand sizes (real finger gun: 0.82-0.84; Sukuna up to 0.43)
    thumb_out=0.65,  # thumb tip away from the index knuckle, in hand sizes (real finger gun: 0.86-0.91; Sukuna up to 0.51)
    cover=0.6,      # both palms within this many FACE widths of the mouth (other real snapshots: 0.73 at the closest)
    reach=1.4,
    fold=1.0,
)
# fingers as landmark chains from the wrist, for drawing the skeleton on the HUD (BGR colour per finger)
BONES = [((0, 1, 2, 3, 4), (200, 200, 200)), ((0, 5, 6, 7, 8), (255, 0, 255)), ((0, 9, 10, 11, 12), (0, 255, 255)),
         ((0, 13, 14, 15, 16), (0, 165, 255)), ((0, 17, 18, 19, 20), (255, 255, 0))]

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_models():
    mdir = os.path.join(HERE, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(mdir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def preflight(model_path):
    """Open a detector in a throwaway subprocess: bad macOS builds abort() uncatchably."""
    code = (
        "import sys\n"
        "from mediapipe.tasks import python as t\n"
        "from mediapipe.tasks.python import vision\n"
        "vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(\n"
        "    base_options=t.BaseOptions(model_asset_path=sys.argv[1]),\n"
        "    running_mode=vision.RunningMode.VIDEO, num_faces=1)).close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, model_path], capture_output=True, text=True)
    if proc.returncode == 0:
        return
    err = (proc.stderr or "") + (proc.stdout or "")
    print(f"\nMediaPipe cannot start a detector here (python {platform.python_version()}, "
          f"mediapipe {getattr(mp, '__version__', '?')}, exit {proc.returncode}).\n")
    if "Service is unavailable" in err or "MetalHelper" in err or proc.returncode == -6:
        print("Cause: mediapipe 0.10.30+ ships macOS wheels that abort on startup.\n"
              "Fix (Python 3.11 or 3.12) — install the pinned set:\n"
              "  pip install -r requirements.txt\n"
              "If you already installed something newer by hand, force it back:\n"
              '  pip install "mediapipe==0.10.21" "numpy<2" "opencv-python<5" "opencv-contrib-python<5"\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return face, hand


class Asset:
    """One reaction: a list of BGRA frames plus per-frame durations (ms) for GIFs."""

    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / h
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            w = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(self.frames[idx], (w, height), interpolation=cv2.INTER_AREA)
        return self._cache[key]


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return Asset([img], [100])


def find_asset_file(pose):
    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        return None
    exts = (".gif", ".png", ".jpg", ".jpeg")
    for fn in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts and (stem == pose or stem.endswith("_" + pose)):
            return os.path.join(adir, fn)
    return None


def load_asset(pose):
    path = find_asset_file(pose)
    if path is None:
        print(f"  {pose:16s} missing -> placeholder")
        return placeholder(pose)
    frames, durations = [], []
    if path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(f.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]
    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)
    print(f"  {pose:16s} {os.path.basename(path)}  ({len(frames)} frame{'s' if len(frames) > 1 else ''})")
    return Asset(frames, durations)


def overlay(frame, sprite, x, y):
    """Alpha-composite BGRA sprite onto BGR frame at top-left (x, y), clipped to the frame."""
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Face:
    """Where the face is and how big (to place the meme), plus the mouth (for higuruma_hiromi)."""

    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        self.box = (int(x0), int(y0), int(x1), int(y1))
        self.w, self.h = float(x1 - x0), float(y1 - y0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.mouth = (p[13] + p[14]) / 2  # between the inner lips


class Hand:
    """Hand-shape features. Landmarks: wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20."""

    def __init__(self, lms, W, H, side=None):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.pts = p
        self.side = side  # MediaPipe's handedness label, "Left" / "Right" (None if unknown); see arms_crossed()
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.size = size = max(dist(p[0], p[9]), 1e-3)  # wrist -> middle knuckle; hand-shape distances are in these units
        # per fingertip (8 index, 12 middle, 16 ring, 20 pinky), kept for the HUD:
        # reach = wrist->tip in hand sizes; fold = wrist->tip / wrist->middle joint; above = tip higher than its knuckle
        tips = (8, 12, 16, 20)
        self.reach = {t: dist(p[0], p[t]) / size for t in tips}
        self.fold = {t: dist(p[0], p[t]) / max(dist(p[0], p[t - 2]), 1e-3) for t in tips}
        self.above = {t: bool(p[t][1] < p[t - 3][1]) for t in tips}
        up = lambda t: self.reach[t] > T["reach"] and self.above[t]  # reaches past the palm, tip above knuckle
        curled = lambda t: self.fold[t] < T["fold"]  # tip folded nearer the wrist than its middle joint
        # cross = middle tip's offset from index tip along the index->middle knuckle line, in knuckle gaps:
        # ~1 side by side, ~0 stacked, <0 crossed over. Knuckles overlapping (hand edge-on) -> can't tell.
        u = p[9] - p[5]
        gap = float(u @ u)
        self.cross = float((p[12] - p[8]) @ u) / gap if gap > (0.1 * size) ** 2 else 1.0
        # Gojo's domain expansion: index + middle up and crossed, ring + pinky folded (thumb is free)
        self.domain = up(8) and up(12) and self.cross < T["cross"] and curled(16) and curled(20)
        # one half of Sukuna's Enmaten seal (middle + ring up, index + pinky curled), as MediaPipe can see it.
        # From the front the hands are side-on: index, middle and ring overlap, and MediaPipe can't tell which of
        # them is curled (it reads the index straight every time). What it does get right is a finger reaching up
        # to the peak, and the curled pinky being short: reach 1.09-1.42 in real snapshots vs 1.66-1.72 in prayer hands.
        self.peak = min((8, 12, 16), key=lambda t: p[t][1])  # highest of the index / middle / ring tips
        self.enmaten = up(self.peak) and self.reach[20] < T["pinky"]
        # one half of Mahito's diamond: a finger up to the peak, pinky pointing down. pinky_drop = how far the pinky tip
        # sits below its knuckle, in hand sizes (+ = below). Middle + ring interlock, so they aren't checked.
        # Real snapshots: -0.43 to -0.03 in Sukuna (curled pinky), about -0.88 in prayer hands (pinky up).
        self.pinky_drop = float(p[20][1] - p[17][1]) / size
        self.mahito = up(self.peak) and self.pinky_drop > T["pinky_drop"]
        # one half of Megumi's Mahoraga summon: a fist. Judged on the four fingers' average fold, so one finger MediaPipe
        # reads loose doesn't flip it (real fists: average 0.76-0.92, but single fingers up to 1.10), plus no finger
        # reaching out from the palm (real fists: longest reach 0.83-0.96; Sukuna's raised finger 1.82+, so a
        # half-curled Sukuna hand, average fold ~1.0, still can't pass as a fist).
        self.fist_mean = sum(self.fold[t] for t in tips) / 4
        self.fist_reach = max(self.reach[t] for t in tips)
        self.fist = self.fist_mean < T["fist"] and self.fist_reach < T["fist_reach"]
        # Hakari: one hand an OK sign (thumb + index tips touching, middle / ring / pinky straight), the other open
        # and flat. Flat = all four fingers straight and not raised: wrist->middle knuckle within ~45 deg of
        # sideways, or pointing down / at the camera. Palm up vs down isn't checked: from a 2D hand that hinges on
        # MediaPipe's left/right label, which is shaky (see arms_crossed).
        straight = lambda t: self.fold[t] > T["straight"]
        self.pinch = dist(p[4], p[8]) / size
        self.ok = self.pinch < T["pinch"] and straight(12) and straight(16) and straight(20)
        self.tilt = float(p[9][1] - p[0][1]) / size
        self.flat_open = all(straight(t) for t in tips) and self.tilt > T["tilt"]
        # Yuta's finger gun: index straight and up, middle / ring / pinky curled, thumb raised and sticking out.
        # The thumb checks keep Sukuna hands out: MediaPipe reads their index straight too, but their thumbs hang low.
        self.index_up = up(8) and straight(8)
        self.gun_curl = max(self.fold[t] for t in (12, 16, 20))
        self.thumb_rise = float(p[2][1] - p[4][1]) / size
        self.thumb_out = dist(p[4], p[5]) / size
        self.gun = (self.index_up and self.gun_curl < T["gun_curl"]
                    and self.thumb_rise > T["thumb_rise"] and self.thumb_out > T["thumb_out"])


def pair_gaps(a, b):
    """Gaps between the two hands, in hand sizes: the closest pair of upright fingertips (index / middle / ring, one
    from each hand; MediaPipe mixes up which is which when fingers overlap), the pinky tips, the wrists, the palms."""
    s = (a.size + b.size) / 2
    tips = min(dist(a.pts[i], b.pts[j]) for i in (8, 12, 16) for j in (8, 12, 16))
    return {"tips": tips / s, "pinky": dist(a.pts[20], b.pts[20]) / s, "wrist": dist(a.pts[0], b.pts[0]) / s,
            "palm": dist(a.palm, b.palm) / s}


def arms_crossed(a, b):
    """True when each hand is on the other's side of the image.
    MediaPipe labels handedness assuming a mirrored (selfie) image. That makes the hand labelled Right sit on the
    image's right with arms uncrossed whether the image is mirrored or not (unmirrored, both the label and the side
    swap), so --no-flip needs no special case. Crossed arms put it left of the hand labelled Left."""
    if {a.side, b.side} != {"Left", "Right"}:
        return False  # unlabelled, or both given the same label: can't tell
    right, left = (a, b) if a.side == "Right" else (b, a)
    return bool(right.palm[0] < left.palm[0])


def hakari_hands(a, b):
    """(OK-sign hand, open hand) if one hand makes the OK sign and the other is open and flat, lower down; else None."""
    for ok, open_ in ((a, b), (b, a)):
        if ok.ok and open_.flat_open and open_.palm[1] > ok.palm[1] + 0.3 * ok.size:
            return ok, open_
    return None


def decide(hands, face=None):
    """Return the first pose that matches, or None. Order is priority."""
    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        g = pair_gaps(a, b)
        # palms not on top of each other: two real hands, not one hand that MediaPipe reported twice
        if a.fist and b.fist and 0.2 < g["palm"] < T["fist_gap"] and arms_crossed(a, b):
            return "megumi_mahoraga"
        if hakari_hands(a, b):
            return "hakari_kinji"
        # the other hand only has to be in frame, so either hand may be the gun
        if a.gun or b.gun:
            return "okkotsu_yuta"
        # wrists apart: two real hands, not one hand that MediaPipe reported twice
        two = g["wrist"] > 0.3 and g["tips"] < T["press"]
        # Mahito first: a down-pointing pinky is short from the wrist too, so a Mahito hand can pass Sukuna's check
        if two and a.mahito and b.mahito and g["pinky"] < T["press"]:
            return "mahito"
        if two and a.enmaten and b.enmaten:
            return "ryomen_sukuna"
        # Higuruma: both palms over the mouth. It checks where the hands are, not their
        # shape, so it goes after the finger-shape signs above.
        if face is not None and max(dist(h.palm, face.mouth) for h in (a, b)) < T["cover"] * face.w:
            return "higuruma_hiromi"
    if any(h.domain for h in hands):
        return "gojo_satoru"
    return None


def finger_line(n, h):
    """One HUD line per hand: the numbers behind the Sukuna and Mahito shape checks, and whether each passes."""
    t = h.peak
    name = {8: "idx", 12: "mid", 16: "ring"}[t]
    up = h.reach[t] > T["reach"] and h.above[t]
    return (f"  hand {n}: peak ({name}) reach {h.reach[t]:.2f}{'' if h.above[t] else ' low'} {'Y' if up else 'n'}   "
            f"pinky reach {h.reach[20]:.2f} {'Y' if h.reach[20] < T['pinky'] else 'n'} (sukuna)   "
            f"pinky drop {h.pinky_drop:+.2f} {'Y' if h.pinky_drop > T['pinky_drop'] else 'n'} (mahito)")


def save_snapshot(img, hand_lms, sides, W, H):
    """Save the preview and every hand's raw landmarks and handedness, to check a rule against real hands."""
    sdir = os.path.join(HERE, "snapshots")
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.join(sdir, time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}")
    cv2.imwrite(stem + ".png", img)
    with open(stem + ".json", "w") as fh:
        json.dump({"W": W, "H": H, "sides": sides,
                   "hands": [[[float(l.x), float(l.y), float(l.z)] for l in h] for h in hand_lms]}, fh)
    print(f"Saved {stem}.png + .json")


def draw_skeleton(img, hands):
    for n, h in enumerate(hands, 1):
        for chain, colour in BONES:
            cv2.polylines(img, [h.pts[list(chain)].astype(np.int32)], False, colour, 2)
        cv2.circle(img, (int(h.palm[0]), int(h.palm[1])), 6, (0, 200, 255), -1)
        cv2.putText(img, f"hand {n} ({h.side or '?'})  cross {h.cross:.2f}", (int(h.pts[0][0]) - 40, int(h.pts[0][1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def draw_hud(img, shown, raw, face, hands):
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    lines = [
        f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {len(hands)}",
        f"keys: q quit  d hud  h skeleton  s snapshot  {' '.join(TEST_KEYS)} test poses",
    ]
    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        g = pair_gaps(a, b)
        yn = lambda ok: "Y" if ok else "n"
        lines[1:1] = [
            f"both: fingertips gap {g['tips']:.2f} (< {T['press']})   wrist {g['wrist']:.2f} (> 0.3)   "
            f"sukuna shape {yn(a.enmaten)}{yn(b.enmaten)}   mahito shape {yn(a.mahito)}{yn(b.mahito)}  "
            f"pinky gap {g['pinky']:.2f} (< {T['press']})",
            f"  peak = highest of index/middle/ring tips, needs reach > {T['reach']} and above its knuckle ('low' if not);"
            f"  pinky reach < {T['pinky']};  pinky drop > {T['pinky_drop']}",
        ] + [finger_line(n, h) for n, h in enumerate((a, b), 1)] + [
            f"megumi: fists {yn(a.fist)}{yn(b.fist)} (avg fold {a.fist_mean:.2f} / {b.fist_mean:.2f} < {T['fist']}, "
            f"longest reach {a.fist_reach:.2f} / {b.fist_reach:.2f} < {T['fist_reach']})   "
            f"palms gap {g['palm']:.2f} (0.2 - {T['fist_gap']})   arms crossed {yn(arms_crossed(a, b))}",
            f"hakari: ok sign {yn(a.ok)}{yn(b.ok)} (thumb-index {a.pinch:.2f} / {b.pinch:.2f} < {T['pinch']})   "
            f"flat open {yn(a.flat_open)}{yn(b.flat_open)} (tilt {a.tilt:+.2f} / {b.tilt:+.2f} > {T['tilt']})   "
            f"open hand lower + matched {yn(hakari_hands(a, b))}",
            f"yuta: finger gun {yn(a.gun)}{yn(b.gun)}   index up {yn(a.index_up)}{yn(b.index_up)}   "
            f"m/r/p fold {a.gun_curl:.2f} / {b.gun_curl:.2f} (< {T['gun_curl']})   "
            f"thumb rise {a.thumb_rise:+.2f} / {b.thumb_rise:+.2f} (> {T['thumb_rise']})   "
            f"out {a.thumb_out:.2f} / {b.thumb_out:.2f} (> {T['thumb_out']})",
            f"higuruma: palms to mouth " + (" / ".join(f"{dist(h.palm, face.mouth) / face.w:.2f}" for h in (a, b))
                                            + f" face widths (< {T['cover']})" if face else "- (no face)"),
            f"  fold idx/mid/ring/pinky  hand 1: " + "/".join(f"{a.fold[t]:.2f}" for t in (8, 12, 16, 20))
            + "   hand 2: " + "/".join(f"{b.fold[t]:.2f}" for t in (8, 12, 16, 20)) + f"   (straight > {T['straight']})",
        ]
    for i, t in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    ap.add_argument("--max-size", type=float, default=MAX_SIZE,
                    help=f"largest the meme gets: its longer side as a fraction of the frame height (default {MAX_SIZE})")
    args = ap.parse_args()

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])
    print("Assets:")
    assets = {pose: load_asset(pose) for pose in POSES}

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() and "x" in args.size:
        w, h = args.size.lower().split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                break
    if not ok:
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - try --camera 1\n"
                 "  - System Settings > Privacy & Security > Camera: allow your terminal app, then re-run")
    H, W = frame.shape[:2]
    print(f"Camera {args.camera}: {W}x{H}")

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet")
        except Exception as e:
            print(f"Virtual camera unavailable ({e}). Preview-only.")

    face_det, hand_det = build_detectors(model_paths)
    shown, hold, show_hud, show_skeleton = None, 0, True, True
    arm = {p: 0 for p in POSES}
    shown_since = 0.0
    forced, forced_until = None, 0.0
    sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
    t_start, last_ts = time.monotonic(), -1
    print(f"Running. Focus the preview window: q quit, d HUD, h skeleton, s snapshot, {' '.join(TEST_KEYS)} test a pose")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if frame.shape[0] != H or frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            ts = int((time.monotonic() - t_start) * 1000)
            ts = last_ts = max(ts, last_ts + 1)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fr = face_det.detect_for_video(mp_img, ts)
            hr = hand_det.detect_for_video(mp_img, ts)
            face = Face(fr.face_landmarks[0], W, H) if fr.face_landmarks else None
            sides = [hd[0].category_name if hd else None for hd in hr.handedness]
            hands = [Hand(h, W, H, s) for h, s in zip(hr.hand_landmarks, sides)]

            raw = decide(hands, face)

            fired = None
            for p in POSES:
                arm[p] = arm[p] + 1 if raw == p else 0
                if raw == p and arm[p] >= ARM.get(p, 3):
                    fired = p
            now = time.monotonic()
            if forced and now < forced_until:
                fired = forced
            if fired:
                if fired != shown:
                    shown_since = now
                shown, hold = fired, HOLD_FRAMES
            elif hold > 0:
                hold -= 1
            else:
                shown = None

            if face is not None:
                sm_center = 0.7 * sm_center + 0.3 * np.array(face.center, np.float32)
                sm_h = 0.7 * sm_h + 0.3 * face.h * FACE_SCALE

            if shown:
                asset = assets[shown]
                idx = asset.frame_at(int((now - shown_since) * 1000))
                longest = args.max_size * H  # longer side: height <= longest, and width (= height * aspect) <= longest
                h = int(min(sm_h, longest, longest / asset.aspect, H * 0.98, (W * 0.98) / asset.aspect)) // 8 * 8
                sprite = asset.scaled(idx, max(h, 8))
                sh, sw = sprite.shape[:2]
                overlay(frame, sprite, int(sm_center[0] - sw / 2), int(sm_center[1] - sh / 2 - 0.05 * sh))

            if vcam:
                vcam.send(frame)
                vcam.sleep_until_next_frame()

            preview = frame.copy() if show_hud or show_skeleton else frame
            if show_skeleton:
                draw_skeleton(preview, hands)
            if show_hud:
                draw_hud(preview, shown, raw, face, hands)
            cv2.imshow("JJK Cam  (q quit, d HUD, h skeleton, s snapshot, number keys test)", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("h"):
                show_skeleton = not show_skeleton
            elif key == ord("s"):
                save_snapshot(preview, hr.hand_landmarks, sides, W, H)
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced, forced_until = POSES[TEST_KEYS.index(chr(key))], now + 2.0
    finally:
        cap.release()
        face_det.close()
        hand_det.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
