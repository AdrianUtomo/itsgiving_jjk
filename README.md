# It's giving... Domain Expansion

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e4efcd4b-3ec8-4287-b38b-cc4aa5036695" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/346406a9-3bdd-4847-8cb1-724a6a38fbd1" width="100%"></td>
  </tr>
</table>

Throw up a Jujutsu Kaisen hand sign at your webcam. It works out *which* sign,
and drops the matching GIF over your face, scaled to follow you around the
frame. You can add your own signs and memes to your heart's desire.

Point Zoom at its virtual camera and the whole call sees your domain expansion.

This project is a fork of [gazijarin's its_giving](https://github.com/gazijarin/itsgiving),
which reacts to facial expressions. This version swaps those for JJK hand signs
and adds per-sign calibration.

```bash
python its_giving_jjk_v2.py --calibrate   # once, about a minute
python its_giving_jjk_v2.py               # preview + virtual camera
python its_giving_jjk_v2.py --no-vcam     # preview only
```

Seven signs: Gojo, Sukuna, Mahito, Megumi, Hakari, Yuta and Higuruma.

There are two files. `its_giving_jjk_v2.py` fits every sign's thresholds to
*your* hands; `its_giving_jjk.py` is the same thing running on fixed numbers
tuned on one person's hands. Use v2.

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 or 3.12. Two MediaPipe models (face and hand, ~11 MB) download
themselves into `models/` on first run.

**Don't unpin the dependencies.** MediaPipe 0.10.30+ (including 1.0.x) ships
macOS wheels that abort the moment they open a detector, so it's held at
0.10.21. That build needs NumPy 1.x, and OpenCV 5 needs NumPy 2 — and 0.10.21
asks for an *unpinned* `opencv-contrib-python`, which quietly drags OpenCV 5 and
therefore NumPy 2 back in. That's why the OpenCV pins are in there even though
nothing in the code cares. Unpin one and you have to unpin all three.

On startup it opens a detector in a throwaway subprocess first, so a broken
MediaPipe build prints the fix instead of crashing. `--skip-check` skips that.

---

## Running it

```bash
python its_giving_jjk_v2.py --calibrate              # all seven signs
python its_giving_jjk_v2.py --calibrate gojo yuta    # just these (short names or prefixes work)
python its_giving_jjk_v2.py
```

| flag | does |
|---|---|
| `--camera N` | webcam index (try `1` if `0` is your phone) |
| `--calibrate [SIGN ...]` | fit thresholds to your hands, save them to `calibration_jjk.json`, exit |
| `--no-calibration` | ignore `calibration_jjk.json` and use the fixed v1 thresholds |
| `--no-vcam` | preview only, no virtual camera |
| `--size WxH` | capture size, default `1280x720` (lower = faster) |
| `--no-flip` | don't mirror the image |
| `--max-size F` | cap the meme's longer side at this fraction of the frame height (default `0.6`, so it stays off your hands) |
| `--skip-check` | skip the MediaPipe startup check |

`its_giving_jjk.py` takes the same flags minus the calibration ones.

| key | does |
|---|---|
| `q` | quit |
| `d` | toggle the HUD |
| `h` | toggle the hand skeleton |
| `c` | recalibrate every sign (v2 only) |
| `s` | save a snapshot — the preview plus raw hand landmarks — to `snapshots/` |
| `1`–`7` | force a sign on screen for 2 seconds |

### Calibrating

For each sign you get 3 seconds to get your hands into place, then 4 seconds
of holding it. Wiggle a little while you hold it, the way you'd really make the
sign, so the thresholds learn your range and not one frozen pose. A reference
thumbnail of the GIF sits in the corner so you know which one you're on.

| key | while calibrating |
|---|---|
| `space` | skip the countdown and start recording now |
| `n` | skip this sign (it keeps its old thresholds) |
| `q` | stop (signs already done are kept) |

Afterwards it prints, per sign, how often the sign passed during your take
before and after calibration, and warns you if something's off — your hand
averaging past a limit, a check that only held in a few frames, or the take
mostly reading as a *different* sign. Redo just that one with
`--calibrate <name>`.

Without a `calibration_jjk.json` v2 runs on the v1 numbers and says so on the
HUD.

---

## Using it in meetings

The virtual camera is on by default, and Zoom, Meet, Teams, Discord and OBS all
treat it as a normal webcam.

**1. Install a backend** (once):

| OS | do this |
|---|---|
| macOS | install [OBS Studio](https://obsproject.com), open it once, quit it |
| Windows | install OBS Studio, or run its virtual-camera installer |
| Linux | `sudo apt install v4l2loopback-dkms` then `sudo modprobe v4l2loopback` |

**2. Run it.** It prints the device it's publishing to:

```
Virtual camera: 'OBS Virtual Camera'  <- pick this camera in Zoom / Meet
```

**3. Pick that device** in your meeting app — Zoom: Settings → Video → Camera.
Meet, Teams and Discord all have the same setting under Video.

**Start this before your meeting app.** Most of them scan for cameras once at
launch and won't notice a device that appeared later.

The meeting sees the clean feed — no HUD, no skeleton. Those only appear in
your preview window.

---

## The signs

| key | pose | the sign | what's actually checked |
|---|---|---|---|
| `1` | `gojo_satoru` | Gojo's Unlimited Void | one hand: index + middle up and crossed, ring + pinky folded |
| `2` | `ryomen_sukuna` | Sukuna's Malevolent Shrine (Enmaten seal) | two hands: upright fingertips touching at a peak, pinkies curled short |
| `3` | `mahito` | Mahito's Self-Embodiment of Perfection | two hands in a diamond: upright fingertips touching on top, pinkies pointing down with their tips touching |
| `4` | `megumi_mahoraga` | Megumi summoning Mahoraga | two fists, close together, arms crossed |
| `5` | `hakari_kinji` | Hakari's Idle Death Gamble | one hand an OK sign, the other open and flat, lower down |
| `6` | `okkotsu_yuta` | Yuta's cursed energy beam | one hand a finger gun (index + thumb up), the other hand anywhere in frame |
| `7` | `higuruma_hiromi` | Higuruma | both palms over your mouth |

A few of these are looser than the real sign, on purpose — they check what
MediaPipe can actually see:

- **Sukuna / Mahito.** From the front your hands are side-on and the index,
  middle and ring fingers overlap. MediaPipe can't tell which of them is
  curled, so it checks for *a* finger reaching up to the peak, and tells the
  two apart by the pinky: curled short for Sukuna, pointing down for Mahito.
- **Megumi.** "Arms crossed" comes from MediaPipe's left/right hand labels:
  each hand has to be on the other one's side of the image.
- **Hakari.** Either hand can be the OK sign. Palm up isn't checked — from a 2D
  hand that hinges on those same left/right labels, which are shaky.

### Assets

Assets live in `assets/`, named after the pose — `gojo_satoru.gif`,
`mahito.gif`. Swap in your own by dropping a file with the right name; GIF,
PNG and JPEG all work, alpha channels composite properly, and GIF frame
timings are read from the file. A `something_` prefix is ignored, so
`2024_gojo_satoru.gif` still counts. A missing asset gets you a red
placeholder, not a crash. Press the sign's number key to check it sits right
on your face.

---

## Adding a sign (v2)

**1.** Drop `assets/todo_aoi.gif` in place.

**2.** Register it. Add `"todo_aoi"` to `POSES`, a key to `TEST_KEYS`
(`"12345678"`), and an entry to `SHORT` (`"todo"`, for `--calibrate todo`),
`HOW` (the one-line instruction shown while calibrating) and `ARM`.

**3.** Add it to `ORDER`. This is the list `decide()` actually walks — first
match wins — so put it above anything it might be mistaken for.

**4.** Give it thresholds in `LIMITS`, each as `(direction, default, (rail, rail))`:

```python
    "todo_aoi": {
        "clap": ("<", 0.5, (0.3, 0.8)),   # palm centre to palm centre, in hand sizes
    },
```

**5.** Write a reader and add it to `READERS`. A reader returns every way the
hands in frame could be making the sign, as `(readings, gates)` — readings
get compared against `LIMITS` and calibrated, gates are plain yes/no checks:

```python
def read_todo(hands, face):
    if len(hands) < 2:
        return []
    a, b = hands[0], hands[1]
    return [({"clap": pair_gaps(a, b)["palm"]}, {"two hands": pair_gaps(a, b)["wrist"] > 0.3})]
```

You have `hands` (`.pts` all 21 landmarks, `.palm`, `.size`, `.side`, and per
fingertip `.reach` / `.fold` / `.above`, plus `.cross`, `.pinky_drop`,
`.pinch`, `.tilt`, `.thumb_rise`, `.thumb_out`, `.fist_mean`, `.fist_reach`),
`face` (`.mouth`, `.center`, `.w`, `.h`), `pair_gaps(a, b)` for the gaps
between two hands, and `arms_crossed(a, b)`.

**6.** Calibrate it (`--calibrate todo`) and tune it against the HUD, which
shows every reading against its threshold and a `Y`/`n` for each. Getting it
to fire is easy; the work is *stopping* it — make every nearby sign and watch
it stay at `no`. Press `s` to save the landmarks of a hand that misbehaves, and
set the rails from real numbers.

### Gotchas

`TEST_KEYS` has one key per pose, matched by position. Adding a sign without a
key is fine (it just gets no test key), but removing one without removing a
key crashes when that key is pressed.

A sign that's in `POSES` but not in `ORDER` never fires.

If a new sign never fires, check `ORDER` before you touch any threshold.
Something earlier matching first is the usual cause, and no threshold can fix
that.

---

## How it works

```
camera frame
     |
 1.  MediaPipe    face: 478 landmarks (where to put the meme, where your mouth is)
                  hands: 2 x 21 points + left/right labels
     |
 2.  Hand         shape measurements, all in hand sizes
     |
 3.  Readers      per sign: readings + gates, for the best way to assign the hands
     |
 4.  decide()     walk ORDER -- first sign whose readings pass YOUR thresholds wins
     |
 5.  arm / hold   must persist 5 frames to fire, lingers 10 frames after
     |
  overlay         scaled to your face, alpha-composited, GIFs animated
```

### Normalising away the camera

Landmarks come out as pixel coordinates, which depend on how far your hands are
from the lens. So nothing is compared in pixels: every hand distance is divided
by that hand's size (wrist to middle knuckle) first. "Fingertips within 0.4"
means within 0.4 hand sizes, and that means the same thing at 40 cm and at a
metre and a half. Higuruma's check is in face widths instead, since it's about
where your hands are relative to your face.

The workhorse is **fold**: wrist-to-fingertip divided by wrist-to-middle-joint.
A straight finger reads about 1.3–1.5; curled, under 1. It doesn't care how
your hand is rotated, which raw angles would.

### Why fixed thresholds don't work, and what to do instead

v1's numbers were tuned on one person's snapshots. Hands aren't the same shape,
nobody makes a sign quite the same way twice, and MediaPipe reads some hands
more cleanly than others. Real fists alone span an average fold of 0.76–0.92,
so any constant is too tight for someone and too loose for someone else.

v2 watches you hold each sign and records the **mean and the standard
deviation** of every reading. Each threshold then sits `K = 3` of *your*
standard deviations out from your average, on the loose side:

```
threshold = your mean ± 3 × your wobble        (clamped to the rails)
```

The rails are what stop it from learning nonsense. Each one stops short of what
a *different* hand shape reads. Sukuna's pinky check, for example, defaults to
reach < 1.55, and a curled Sukuna pinky reads 1.09–1.42 while prayer hands read
1.66–1.72 — so the rail is 1.62, and however you calibrate it, prayer hands
never set off Sukuna. If your own average lands past a rail, calibration says
so rather than quietly accepting a sign that'll never fire.

There's also a floor of 0.04 on the standard deviation, so a dead-still take
can't park a threshold right on top of your average.

---

## What's where

```
its_giving_jjk_v2.py   the calibrated version — the one to use
its_giving_jjk.py      v1: same signs, fixed thresholds
calibration_jjk.json   your thresholds, sign by sign (made by --calibrate)
requirements.txt       pinned on purpose — read the comments before changing them
assets/                the GIFs, named after their pose
models/                MediaPipe .task files (downloaded on first run)
snapshots/             PNG + landmark JSON from the s key
```

Inside v2: `POSES` / `ORDER` / `LIMITS` / `ARM` is the tuning block, `Hand`
turns 21 landmarks into shape measurements, the `read_*()` functions turn
those into each sign's readings, `decide()` walks `ORDER`, and `fit()` /
`run_calibration()` are the hold-each-sign session.

Want to change **what sets off what**? The `read_*()` functions and `ORDER`.
Want to change **how easily it goes off**? `LIMITS` (defaults and rails), `K`
and `ARM` — or just recalibrate.
