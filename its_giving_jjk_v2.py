#!/usr/bin/env python3
"""
its_giving_jjk_v2.py — Jujutsu Kaisen hand signs on top of your face, live in Zoom / Meet.

Same as its_giving_jjk.py, but every sign's thresholds are fitted to your own hands instead of to constants tuned
on one person's snapshots. Calibration has you hold each sign for a few seconds; each threshold then sits K of your
standard deviations out from where your hand actually was, clamped between rails that keep it clear of what other
hand shapes read. Uncalibrated, it decides exactly like its_giving_jjk.py. See README.md.

  python its_giving_jjk_v2.py --calibrate [gojo sukuna ...]     (no names = all seven, in order)
  python its_giving_jjk_v2.py [--camera 1] [--no-vcam] [--size 640x480] [--no-flip] [--max-size 0.6]

Keys:  q quit   d toggle HUD   h toggle hand skeleton   c recalibrate every sign
       s save a snapshot (preview + hand landmarks) to snapshots/   1-7 force-show a pose
While calibrating:  space start now   n skip this sign (keeps its old thresholds)   q stop (keeps the signs done)

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
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

POSES = ["gojo_satoru", "ryomen_sukuna", "mahito", "megumi_mahoraga", "hakari_kinji", "okkotsu_yuta", "higuruma_hiromi"]
TEST_KEYS = "1234567"
SHORT = {"gojo_satoru": "gojo", "ryomen_sukuna": "sukuna", "mahito": "mahito", "megumi_mahoraga": "megumi",
         "hakari_kinji": "hakari", "okkotsu_yuta": "yuta", "higuruma_hiromi": "higuruma"}
# decide() checks in this order and the first match wins. Finger shapes go before Higuruma (it checks where the hands
# are, not their shape), and Mahito before Sukuna (a down-pointing pinky is short from the wrist too, so a Mahito hand
# can pass Sukuna's check).
ORDER = ["megumi_mahoraga", "hakari_kinji", "okkotsu_yuta", "mahito", "ryomen_sukuna", "higuruma_hiromi", "gojo_satoru"]
HOW = {
    "gojo_satoru": "one hand: index + middle up and crossed, ring + pinky folded",
    "ryomen_sukuna": "two hands: middle + ring up, tips meeting at a peak, index + pinky curled",
    "mahito": "two hands in a diamond: index tips together on top, pinky tips together below",
    "megumi_mahoraga": "two fists, overlapping, arms crossed",
    "hakari_kinji": "one hand an OK sign, the other open and flat, lower down",
    "okkotsu_yuta": "one hand a finger gun (index + thumb up), the other hand in frame",
    "higuruma_hiromi": "both hands over your mouth",
}

FACE_SCALE = 2.0
MAX_SIZE = 0.6  # cap on the meme's longer side, as a fraction of the frame height, so it stays off your hands
                # (a 16:9 GIF at 0.6 still covers the face but ends about at the chin; uncapped it hung ~80px below)
HOLD_FRAMES = 10
ARM = {"gojo_satoru": 5, "ryomen_sukuna": 5, "mahito": 5, "megumi_mahoraga": 5, "hakari_kinji": 5, "okkotsu_yuta": 5,
       "higuruma_hiromi": 5}

# Every threshold, sign by sign: name -> (direction, default, (rail, rail)). "<" passes below the threshold, ">" above.
# Distances are in hand sizes (wrist -> middle knuckle) unless noted; fold = wrist->tip / wrist->middle joint, so
# under 1 is curled. Defaults are its_giving_jjk.py's constants. Calibration moves each one to your hands but never
# past its rails, which stop short of what other hand shapes read (bracketed ranges are from real snapshots).
LIMITS = {
    "gojo_satoru": {
        "up": (">", 1.4, (1.2, 1.6)),        # index + middle tip reach, the shorter of the two (fists [0.83-0.96])
        "cross": ("<", 0.25, (-0.1, 0.45)),  # middle tip along the index->middle knuckle line: ~1 side by side,
                                             # ~0 stacked, <0 crossed over
        "curl": ("<", 1.0, (0.85, 1.1)),     # ring + pinky fold, the looser of the two (straight [1.29-1.47])
    },
    "ryomen_sukuna": {
        "up": (">", 1.4, (1.2, 1.6)),        # each hand's highest index / middle / ring tip, the shorter (Sukuna [1.82+])
        "pinky": ("<", 1.55, (1.3, 1.62)),   # pinky reach, the longer (Sukuna [1.09-1.42], prayer hands [1.66-1.72])
        "press": ("<", 0.4, (0.2, 0.6)),     # closest pair of upright fingertips, one from each hand
    },
    "mahito": {
        "up": (">", 1.4, (1.2, 1.6)),
        "drop": (">", 0.15, (0.0, 0.4)),     # pinky tip below its knuckle, the higher of the two
                                             # (Sukuna's curled pinky [-0.43 to -0.03], prayer hands ~-0.88)
        "press": ("<", 0.4, (0.2, 0.6)),
        "pinkies": ("<", 0.4, (0.2, 0.6)),   # pinky tip to pinky tip
    },
    "megumi_mahoraga": {
        "fist": ("<", 1.0, (0.85, 1.05)),    # the four fingers' average fold, looser hand (fists [0.76-0.92], though
                                             # single fingers read up to 1.10; half-curled Sukuna ~1.0)
        "reach": ("<", 1.3, (1.0, 1.5)),     # longest fingertip reach, either hand (fists [0.83-0.96]; Sukuna's raised
                                             # finger [1.82+], so a half-curled Sukuna hand still can't pass as a fist)
        "gap": ("<", 1.8, (1.4, 2.2)),       # palm centre to palm centre (fists side by side [1.39-1.50])
    },
    "hakari_kinji": {
        "pinch": ("<", 0.3, (0.15, 0.45)),       # OK hand: thumb tip to index tip
        "ok_fingers": (">", 1.15, (1.05, 1.25)),  # OK hand: middle / ring / pinky fold, the most bent
                                                  # (straight [1.29-1.47], curled < 1.0)
        "flat": (">", 1.15, (1.05, 1.25)),       # open hand: all four fingers' fold, the most bent
        "tilt": (">", -0.7, (-0.9, -0.4)),       # open hand's wrist->middle knuckle, vertical part: -1 fingers up,
                                                 # 0 sideways, +1 down
        "lower": (">", 0.3, (0.15, 0.8)),        # open palm below the OK palm, in the OK hand's sizes
    },
    "okkotsu_yuta": {
        "up": (">", 1.4, (1.2, 1.6)),            # gun hand: index reach
        "straight": (">", 1.15, (1.05, 1.25)),   # gun hand: index fold
        "curl": ("<", 1.1, (0.9, 1.2)),          # middle / ring / pinky fold, the least bent (finger gun [0.80-1.01])
        "thumb_up": (">", 0.6, (0.48, 0.75)),    # thumb tip above its base joint (finger gun [0.82-0.84], Sukuna up to 0.43)
        "thumb_out": (">", 0.65, (0.55, 0.8)),   # thumb tip away from the index knuckle
                                                 # (finger gun [0.86-0.91], Sukuna up to 0.51)
    },
    "higuruma_hiromi": {
        "cover": ("<", 0.6, (0.4, 0.68)),        # the farther palm to the mouth, in FACE widths
                                                 # (other real snapshots: 0.73 at the closest)
    },
}

CALIB_FILE = "calibration_jjk.json"
CALIB_VERSION = 1
READY_SECONDS = 3.0    # countdown before each sign, to get your hands into place
CALIB_SECONDS = 4.0    # then this long holding it
CALIB_MIN_SAMPLES = 20
K = 3.0                # a threshold sits this many of your standard deviations out from your average, on the loose side
SIGMA_FLOOR = 0.04     # so a dead-still take can't park a threshold right on top of your average

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


class Clock:
    """Strictly increasing timestamps for the life of a detector, recalibrations included."""

    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


class Profile:
    """Your thresholds, sign by sign. A sign you haven't calibrated keeps its its_giving_jjk.py defaults."""

    def __init__(self, signs=None):
        self.signs = signs or {}  # pose -> {"made", "samples", "mean", "sigma", "t"}

    def t(self, pose, key):
        v = self.signs.get(pose, {}).get("t", {}).get(key)
        return LIMITS[pose][key][1] if v is None else v

    def calibrated(self, pose):
        return pose in self.signs

    @property
    def generic(self):
        return not self.signs

    def with_sign(self, pose, entry):
        return Profile(dict(self.signs, **{pose: entry}))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"version": CALIB_VERSION, "signs": self.signs}, fh, indent=1, sort_keys=True)

    @staticmethod
    def load(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return Profile()
        if not isinstance(data, dict) or data.get("version") != CALIB_VERSION or not isinstance(data.get("signs"), dict):
            return Profile()
        signs = {}
        for pose, e in data["signs"].items():
            try:
                # re-clamp: the file may be hand-edited, or older than the current rails
                t = {k: clamp(float(v), *LIMITS[pose][k][2]) for k, v in e["t"].items() if k in LIMITS[pose]}
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            signs[pose] = dict(e, t=t)
        return Profile(signs)


GENERIC = Profile()


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
    """Hand-shape measurements; the sign checks that use them are the read_*() functions and LIMITS.
    Landmarks: wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20."""

    def __init__(self, lms, W, H, side=None):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.pts = p
        self.side = side  # MediaPipe's handedness label, "Left" / "Right" (None if unknown); see arms_crossed()
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.size = size = max(dist(p[0], p[9]), 1e-3)  # wrist -> middle knuckle; hand-shape distances are in these units
        # per fingertip (8 index, 12 middle, 16 ring, 20 pinky):
        # reach = wrist->tip in hand sizes; fold = wrist->tip / wrist->middle joint; above = tip higher than its knuckle
        tips = (8, 12, 16, 20)
        self.reach = {t: dist(p[0], p[t]) / size for t in tips}
        self.fold = {t: dist(p[0], p[t]) / max(dist(p[0], p[t - 2]), 1e-3) for t in tips}
        self.above = {t: bool(p[t][1] < p[t - 3][1]) for t in tips}
        # cross = middle tip's offset from index tip along the index->middle knuckle line, in knuckle gaps:
        # ~1 side by side, ~0 stacked, <0 crossed over. Knuckles overlapping (hand edge-on) -> can't tell.
        u = p[9] - p[5]
        gap = float(u @ u)
        self.cross = float((p[12] - p[8]) @ u) / gap if gap > (0.1 * size) ** 2 else 1.0
        # Sukuna / Mahito: from the front the hands are side-on, index, middle and ring overlap, and MediaPipe can't
        # tell which of them is curled (it reads the index straight every time). What it does get right is a finger
        # reaching up to the peak, and the pinky: curled short for Sukuna, pointing down for Mahito.
        self.peak = min((8, 12, 16), key=lambda t: p[t][1])  # highest of the index / middle / ring tips
        self.pinky_drop = float(p[20][1] - p[17][1]) / size  # pinky tip below its knuckle (+ = below)
        # Megumi: judged on the four fingers' average fold, so one finger MediaPipe reads loose doesn't flip it,
        # plus no finger reaching out from the palm
        self.fist_mean = sum(self.fold[t] for t in tips) / 4
        self.fist_reach = max(self.reach[t] for t in tips)
        # Hakari: OK sign = thumb + index tips touching; the open hand's tilt says raised vs flat. Palm up vs down isn't
        # checked: from a 2D hand that hinges on MediaPipe's left/right label, which is shaky (see arms_crossed).
        self.pinch = dist(p[4], p[8]) / size
        self.tilt = float(p[9][1] - p[0][1]) / size
        # Yuta: the thumb checks keep Sukuna hands out; MediaPipe reads their index straight too, but their thumbs hang low
        self.gun_curl = max(self.fold[t] for t in (12, 16, 20))
        self.thumb_rise = float(p[2][1] - p[4][1]) / size
        self.thumb_out = dist(p[4], p[5]) / size


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


# Readers: every way the hands in frame could be making a sign, as a list of (readings, gates). Readings are compared
# against that sign's LIMITS; gates are yes/no conditions with nothing to calibrate. One entry per role assignment
# (which hand is the OK sign, which is the gun); empty when there aren't enough hands, or no face, to read the sign.

def read_gojo(hands, face):
    return [({"up": min(h.reach[8], h.reach[12]), "cross": h.cross, "curl": max(h.fold[16], h.fold[20])},
             {"tips above knuckles": h.above[8] and h.above[12]}) for h in hands]


def read_sukuna(hands, face):
    if len(hands) < 2:
        return []
    a, b = hands[0], hands[1]
    g = pair_gaps(a, b)
    return [({"up": min(a.reach[a.peak], b.reach[b.peak]), "pinky": max(a.reach[20], b.reach[20]), "press": g["tips"]},
             # wrists apart: two real hands, not one hand that MediaPipe reported twice
             {"peaks above knuckles": a.above[a.peak] and b.above[b.peak], "two hands": g["wrist"] > 0.3})]


def read_mahito(hands, face):
    if len(hands) < 2:
        return []
    a, b = hands[0], hands[1]
    g = pair_gaps(a, b)
    return [({"up": min(a.reach[a.peak], b.reach[b.peak]), "drop": min(a.pinky_drop, b.pinky_drop),
              "press": g["tips"], "pinkies": g["pinky"]},
             {"peaks above knuckles": a.above[a.peak] and b.above[b.peak], "two hands": g["wrist"] > 0.3})]


def read_megumi(hands, face):
    if len(hands) < 2:
        return []
    a, b = hands[0], hands[1]
    g = pair_gaps(a, b)
    return [({"fist": max(a.fist_mean, b.fist_mean), "reach": max(a.fist_reach, b.fist_reach), "gap": g["palm"]},
             # palms not on top of each other: two real hands, not one hand that MediaPipe reported twice
             {"arms crossed": arms_crossed(a, b), "two hands": g["palm"] > 0.2})]


def read_hakari(hands, face):
    if len(hands) < 2:
        return []
    return [({"pinch": ok.pinch, "ok_fingers": min(ok.fold[t] for t in (12, 16, 20)),
              "flat": min(open_.fold[t] for t in (8, 12, 16, 20)), "tilt": open_.tilt,
              "lower": float(open_.palm[1] - ok.palm[1]) / ok.size}, {})
            for ok, open_ in ((hands[0], hands[1]), (hands[1], hands[0]))]


def read_yuta(hands, face):
    if len(hands) < 2:
        return []  # the other hand only has to be in frame, so either hand may be the gun
    return [({"up": h.reach[8], "straight": h.fold[8], "curl": h.gun_curl, "thumb_up": h.thumb_rise,
              "thumb_out": h.thumb_out}, {"index above knuckle": h.above[8]}) for h in hands[:2]]


def read_higuruma(hands, face):
    if len(hands) < 2 or face is None:
        return []
    return [({"cover": max(dist(h.palm, face.mouth) for h in hands[:2]) / face.w}, {})]


READERS = {"gojo_satoru": read_gojo, "ryomen_sukuna": read_sukuna, "mahito": read_mahito,
           "megumi_mahoraga": read_megumi, "hakari_kinji": read_hakari, "okkotsu_yuta": read_yuta,
           "higuruma_hiromi": read_higuruma}


def margins(pose, values, prof):
    """How far each reading is on the passing side of its threshold (negative = failing)."""
    out = {}
    for key, v in values.items():
        thr = prof.t(pose, key)
        out[key] = thr - v if LIMITS[pose][key][0] == "<" else v - thr
    return out


def read(pose, hands, face, prof):
    """(passes, readings, gates) for the role assignment that comes closest to the sign; None if it can't be read."""
    top = None
    for values, gates in READERS[pose](hands, face):
        m = margins(pose, values, prof)
        ok = all(x > 0 for x in m.values()) and all(gates.values())
        score = (ok, sum(x > 0 for x in m.values()) + sum(map(bool, gates.values())), sum(min(x, 0.5) for x in m.values()))
        if top is None or score > top[0]:
            top = (score, values, gates)
    return None if top is None else (top[0][0], top[1], top[2])


def decide(hands, face, prof):
    """Return the first pose that matches, or None."""
    for pose in ORDER:
        r = read(pose, hands, face, prof)
        if r and r[0]:
            return pose
    return None


def fit(pose, frames):
    """One sign's thresholds from a take of you holding it: K sigma out from your average, then clamped to the rails.
    Roles (which hand is which) are picked under the defaults, so an earlier calibration can't skew a new one."""
    reads = [read(pose, hands, face, GENERIC)[1] for hands, face in frames]
    mean, sigma, t = {}, {}, {}
    for key, (op, _, rails) in LIMITS[pose].items():
        xs = np.array([r[key] for r in reads], np.float64)
        m, s = float(xs.mean()), max(float(xs.std()), SIGMA_FLOOR)
        mean[key], sigma[key] = round(m, 4), round(s, 4)
        t[key] = round(clamp(m + K * s if op == "<" else m - K * s, *rails), 3)
    return {"made": time.strftime("%Y-%m-%d %H:%M"), "samples": len(reads), "mean": mean, "sigma": sigma, "t": t}


def pass_rate(pose, frames, prof):
    return sum(bool(read(pose, h, f, prof)[0]) for h, f in frames) / len(frames)


def report(takes, old, new):
    """Print what each take taught us, and catch the ways it can still go wrong."""
    for pose, frames in takes.items():
        e = new.signs[pose]
        print(f"\n{pose}: {e['samples']} frames. The sign passes in {pass_rate(pose, frames, new):.0%} of them "
              f"(was {pass_rate(pose, frames, old):.0%}).")
        warn = []
        for key, (op, default, rails) in LIMITS[pose].items():
            m, s, thr = e["mean"][key], e["sigma"][key], e["t"][key]
            print(f"    {key:11s} {m:+.2f} ± {s:.2f}    {op} {default:.2f} -> {thr:.2f}"
                  f"{'  (at the rail)' if thr in rails else ''}")
            if (m >= thr) if op == "<" else (m <= thr):
                warn.append(f"your {key} averages {m:.2f}, past the {thr:.2f} limit — beyond it the sign reads like "
                            "a different hand shape, so this one will rarely fire. Adjust your hand and redo it")
        gates = [read(pose, h, f, new)[2] for h, f in frames]
        for g in gates[0]:
            held = sum(bool(x[g]) for x in gates) / len(gates)
            if held < 0.5:
                warn.append(f"'{g}' only held in {held:.0%} of frames")
        wins = Counter(decide(h, f, new) for h, f in frames)
        for other, n in wins.most_common():
            if other not in (pose, None) and n / len(frames) > 0.2:
                warn.append(f"{n / len(frames):.0%} of these frames come out as {other} instead")
        for w in warn:
            print(f"    ! {w}")


def grab(cap, dets, clock, args, W, H):
    """Read a frame, mirror it, run both detectors. Returns (frame, face, hands, raw hand landmarks, sides), or None."""
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.shape[0] != H or frame.shape[1] != W:
        frame = cv2.resize(frame, (W, H))
    if not args.no_flip:
        frame = cv2.flip(frame, 1)
    ts = clock.next()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fr = dets[0].detect_for_video(mp_img, ts)
    hr = dets[1].detect_for_video(mp_img, ts)
    face = Face(fr.face_landmarks[0], W, H) if fr.face_landmarks else None
    sides = [hd[0].category_name if hd else None for hd in hr.handedness]
    hands = [Hand(h, W, H, s) for h, s in zip(hr.hand_landmarks, sides)]
    return frame, face, hands, hr.hand_landmarks, sides


def hold_sign(cap, dets, clock, args, W, H, window, pose, n, total, prof, assets):
    """Count down, then record you holding one sign. Returns its frames as (hands, face) pairs, "skip" or "stop"."""
    frames, start, recording = [], time.monotonic(), False
    while True:
        now = time.monotonic()
        if not recording and now - start > READY_SECONDS:
            recording, start = True, now
        if recording and now - start > CALIB_SECONDS:
            return frames
        got = grab(cap, dets, clock, args, W, H)
        if got is None:
            return "stop"
        frame, face, hands, _, _ = got
        r = read(pose, hands, face, prof)
        if recording and r is not None:
            frames.append((hands, face))
        left = (CALIB_SECONDS if recording else READY_SECONDS) - (now - start)
        draw_calibration(frame, pose, n, total, recording, left, len(frames), r, prof, hands, assets[pose])
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "stop"
        if key == ord("n"):
            return "skip"
        if key == ord(" ") and not recording:
            recording, start = True, time.monotonic()


def run_calibration(cap, dets, clock, args, W, H, window, prof, poses, assets):
    """Hold each sign in turn and fit its thresholds to your hands. Returns the updated profile (signs skipped or
    failed keep what they had), or None if no sign was calibrated."""
    print(f"\nCalibrating {len(poses)} sign{'s' if len(poses) > 1 else ''}. For each: {READY_SECONDS:.0f}s to get "
          f"into it, then hold it for {CALIB_SECONDS:.0f}s.\nWiggle a little while you hold it — the way you'd "
          "really make it — so the thresholds learn your range, not one frozen pose.\n"
          "space = start now, n = skip a sign, q = stop")
    takes = {}
    for n, pose in enumerate(poses, 1):
        frames = hold_sign(cap, dets, clock, args, W, H, window, pose, n, len(poses), prof, assets)
        if frames == "stop":
            print("Calibration stopped.")
            break
        if frames == "skip":
            print(f"  {pose}: skipped")
        elif len(frames) < CALIB_MIN_SAMPLES:
            print(f"  {pose}: only {len(frames)} usable frames, keeping its old thresholds — "
                  f"{'get a hand' if pose == 'gojo_satoru' else 'get both hands'}"
                  f"{' and your face' if pose == 'higuruma_hiromi' else ''} in frame and redo it")
        else:
            takes[pose] = frames
    if not takes:
        return None
    new = prof
    for pose, frames in takes.items():
        new = new.with_sign(pose, fit(pose, frames))
    report(takes, prof, new)
    return new


def text(img, t, org, scale=0.55, colour=(0, 255, 0), thick=1):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def draw_calibration(img, pose, n, total, recording, left, samples, r, prof, hands, asset):
    H, W = img.shape[:2]
    draw_skeleton(img, hands)
    cv2.rectangle(img, (0, 0), (W, 112), (0, 0, 0), -1)
    text(img, f"CALIBRATING {n}/{total}: {pose}", (16, 30), 0.8, (255, 255, 255), 2)
    text(img, HOW[pose], (16, 58), 0.55, (200, 200, 200))
    if not recording:
        msg, colour = f"get ready  {left:0.1f}s    space = start now   n = skip   q = stop", (255, 255, 255)
    elif r is None:
        need = ("a hand" if pose == "gojo_satoru" else "both hands") + (" + face" if pose == "higuruma_hiromi" else "")
        msg, colour = f"HOLD IT  {left:0.1f}s   {samples} frames   NEED {need.upper()}", (0, 140, 255)
    else:
        msg, colour = f"HOLD IT  {left:0.1f}s   {samples} frames", (0, 255, 0)
    text(img, msg, (16, 88), 0.6, colour, 2)
    if recording:
        cv2.rectangle(img, (0, 100), (int(W * min(1 - left / CALIB_SECONDS, 1.0)), 112), (0, 220, 0), -1)
    text(img, sign_line(pose, r, prof), (10, 136), 0.5, (0, 255, 255) if r and r[0] else (0, 255, 0))
    thumb = asset.scaled(0, 110)
    overlay(img, thumb, W - thumb.shape[1] - 10, 150)


def sign_line(pose, r, prof):
    """One HUD line: each reading against its threshold (* = calibrated to you), and the gates."""
    name = SHORT[pose] + ("*" if prof.calibrated(pose) else "")
    if r is None:
        return f"{name:9s}  -"
    ok, values, gates = r
    m = margins(pose, values, prof)
    yn = lambda b: "Y" if b else "n"
    parts = [f"{k} {v:.2f}{LIMITS[pose][k][0]}{prof.t(pose, k):.2f} {yn(m[k] > 0)}" for k, v in values.items()]
    parts += [f"{g} {yn(v)}" for g, v in gates.items()]
    return f"{name:9s} {'YES' if ok else 'no '}  " + "   ".join(parts)


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


def draw_hud(img, shown, raw, face, hands, prof):
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    lines = [(f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {len(hands)}", (0, 255, 0))]
    for pose in ORDER:
        r = read(pose, hands, face, prof)
        lines.append((sign_line(pose, r, prof), (0, 255, 255) if r and r[0] else (0, 255, 0)))
    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        lines.append((f"fold idx/mid/ring/pinky  hand 1: " + "/".join(f"{a.fold[t]:.2f}" for t in (8, 12, 16, 20))
                      + "   hand 2: " + "/".join(f"{b.fold[t]:.2f}" for t in (8, 12, 16, 20)), (0, 255, 0)))
    done = sum(prof.calibrated(p) for p in POSES)
    lines += [
        (("NOT CALIBRATED - its_giving_jjk.py's thresholds. press 'c'" if prof.generic else
          f"calibrated {done}/{len(POSES)} signs   (* = your thresholds)"),
         (0, 140, 255) if done < len(POSES) else (200, 200, 200)),
        (f"keys: q quit  d hud  h skeleton  c recalibrate  s snapshot  {' '.join(TEST_KEYS)} test poses", (0, 255, 0)),
    ]
    for i, (t, colour) in enumerate(lines):
        text(img, t, (10, 24 + 22 * i), 0.5, colour)


def pick_signs(names):
    """--calibrate's arguments -> poses, in POSES order. Full names, short names (gojo, yuta) or prefixes all work."""
    if not names:
        return list(POSES)
    out = set()
    for n in names:
        n = n.lower()
        hits = [p for p in POSES if n in (p, SHORT[p])] or [p for p in POSES if p.startswith(n) or SHORT[p].startswith(n)]
        if len(hits) != 1:
            sys.exit(f"Unknown sign '{n}'. Pick from: {', '.join(SHORT.values())}")
        out.add(hits[0])
    return [p for p in POSES if p in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--calibrate", nargs="*", metavar="SIGN",
                    help="fit thresholds to your hands (all signs, or just the ones named), save them, and exit")
    ap.add_argument("--no-calibration", action="store_true", help=f"ignore {CALIB_FILE}; use its_giving_jjk.py's thresholds")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    ap.add_argument("--max-size", type=float, default=MAX_SIZE,
                    help=f"largest the meme gets: its longer side as a fraction of the frame height (default {MAX_SIZE})")
    args = ap.parse_args()
    calibrating = args.calibrate is not None
    to_calibrate = pick_signs(args.calibrate) if calibrating else []

    calib_path = os.path.join(HERE, CALIB_FILE)
    prof = Profile() if args.no_calibration else Profile.load(calib_path)
    if prof.generic and not calibrating and not args.no_calibration:
        print(f"No usable {CALIB_FILE}. Running on its_giving_jjk.py's thresholds, tuned on someone else's hands.\n"
              f"Run:  python {os.path.basename(__file__)} --calibrate")

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

    clock = Clock()
    window = "JJK Cam v2  (q quit, d HUD, h skeleton, c recalibrate, s snapshot, number keys test)"
    dets = build_detectors(model_paths)

    if calibrating:
        try:
            new = run_calibration(cap, dets, clock, args, W, H, window, prof, to_calibrate, assets)
        finally:
            for det in dets:
                det.close()
            cap.release()
            cv2.destroyAllWindows()
        if new is None:
            sys.exit(1)
        new.save(calib_path)
        print(f"\nSaved {CALIB_FILE}. Now run:  python {os.path.basename(__file__)}")
        return

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet")
        except Exception as e:
            print(f"Virtual camera unavailable ({e}). Preview-only.")

    shown, hold, show_hud, show_skeleton = None, 0, True, True
    arm = {p: 0 for p in POSES}
    shown_since = 0.0
    forced, forced_until = None, 0.0
    sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
    print(f"Running. Focus the preview window: q quit, d HUD, h skeleton, c recalibrate, s snapshot, "
          f"{' '.join(TEST_KEYS)} test a pose")

    try:
        while True:
            got = grab(cap, dets, clock, args, W, H)
            if got is None:
                print("Camera stopped returning frames.")
                break
            frame, face, hands, hand_lms, sides = got

            raw = decide(hands, face, prof)

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
                draw_hud(preview, shown, raw, face, hands, prof)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("h"):
                show_skeleton = not show_skeleton
            elif key == ord("s"):
                save_snapshot(preview, hand_lms, sides, W, H)
            elif key == ord("c"):
                new = run_calibration(cap, dets, clock, args, W, H, window, prof, list(POSES), assets)
                if new is not None:
                    prof = new
                    prof.save(calib_path)
                    print(f"Saved {CALIB_FILE}.")
                shown, hold, forced = None, 0, None
                arm = {p: 0 for p in POSES}
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced, forced_until = POSES[TEST_KEYS.index(chr(key))], now + 2.0
    finally:
        cap.release()
        for det in dets:
            det.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
