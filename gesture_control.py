"""
╔══════════════════════════════════════════════════════════════╗
║          Hand Gesture Recognition – Demo Script             ║
║   MediaPipe Tasks API + OpenCV  |  Real-time Webcam         ║
╚══════════════════════════════════════════════════════════════╝

Gestures recognized:
  ✋  Open Palm     → ▶  Play
  ✊  Fist          → ⏸  Pause
  👍  Thumbs Up    → 🔊  Volume Up
  👎  Thumbs Down  → 🔉  Volume Down
  ☝️   One Finger   → ⏭  Next Track
  ✌️   Two Fingers  → ⏮  Previous Track
  🤟  Three Fingers → 🔀  Shuffle
  🤙  Pinky Only   → 🔇  Mute

─────────────────────────────────────────────────────────────
SETUP (one-time):
  pip install mediapipe opencv-python numpy

  # Download the hand landmark model (~8 MB):
  curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o hand_landmarker.task

  # OR let the script auto-download on first run.

USAGE:
  python hand_gesture_demo.py
  python hand_gesture_demo.py --model /path/to/hand_landmarker.task
  python hand_gesture_demo.py --camera 1   # use camera index 1

CONTROLS (while running):
  Q  or  ESC  →  Quit
  S           →  Save screenshot
─────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("❌  mediapipe not found.  Run:  pip install mediapipe")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
#  Landmark Indices  (MediaPipe 21-point hand model)
# ──────────────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC=1;  THUMB_MCP=2;  THUMB_IP=3;   THUMB_TIP=4
INDEX_MCP=5;  INDEX_PIP=6;  INDEX_DIP=7;  INDEX_TIP=8
MIDDLE_MCP=9; MIDDLE_PIP=10; MIDDLE_DIP=11; MIDDLE_TIP=12
RING_MCP=13;  RING_PIP=14;  RING_DIP=15;  RING_TIP=16
PINKY_MCP=17; PINKY_PIP=18; PINKY_DIP=19; PINKY_TIP=20

# Skeleton connectivity
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]


# ──────────────────────────────────────────────────────────────
#  Gesture Classifier
# ──────────────────────────────────────────────────────────────
def finger_states(lm: list, handedness: str) -> list:
    """
    Returns [thumb, index, middle, ring, pinky] extended states (bool).
    Uses y-axis for fingers, x-axis for thumb (mirrored for left/right).
    """
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pips = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    ext  = []

    # Thumb: horizontal comparison (flipped per hand)
    if handedness == "Right":
        ext.append(lm[THUMB_TIP].x < lm[THUMB_IP].x)
    else:
        ext.append(lm[THUMB_TIP].x > lm[THUMB_IP].x)

    # Other fingers: tip above pip (smaller y = higher on screen)
    for tip, pip in zip(tips[1:], pips[1:]):
        ext.append(lm[tip].y < lm[pip].y)

    return ext


def classify_gesture(lm: list, handedness: str) -> tuple:
    """Returns (gesture label, mapped action)."""
    thumb, index, middle, ring, pinky = finger_states(lm, handedness)

    # Thumbs Up / Down  (only thumb extended)
    if thumb and not index and not middle and not ring and not pinky:
        if lm[THUMB_TIP].y < lm[WRIST].y:
            return "👍 Thumbs Up",     "🔊  Volume Up"
        else:
            return "👎 Thumbs Down",   "🔉  Volume Down"

    # Open Palm  (all extended)
    if thumb and index and middle and ring and pinky:
        return "✋ Open Palm",      "▶   Play"

    # Fist  (all closed)
    if not thumb and not index and not middle and not ring and not pinky:
        return "✊ Fist",           "⏸   Pause"

    # Three Fingers
    if not thumb and index and middle and ring and not pinky:
        return "🤟 Three Fingers",  "🔀   Shuffle"

    # Two Fingers / Peace
    if not thumb and index and middle and not ring and not pinky:
        return "✌️  Two Fingers",   "⏮   Prev Track"

    # One Finger
    if not thumb and index and not middle and not ring and not pinky:
        return "☝️  One Finger",    "⏭   Next Track"

    # Pinky Only
    if not thumb and not index and not middle and not ring and pinky:
        return "🤙 Pinky",          "🔇   Mute"

    return "❓ Unknown", "—"


# ──────────────────────────────────────────────────────────────
#  Action Logger
# ──────────────────────────────────────────────────────────────
class ActionLog:
    def __init__(self, maxlen: int = 8, cooldown: float = 1.5):
        self._log         = deque(maxlen=maxlen)
        self._last_action = ""
        self._last_time   = 0.0
        self.cooldown     = cooldown

    def push(self, gesture: str, action: str):
        now = time.time()
        if action == "—":
            return
        if action != self._last_action or (now - self._last_time) > self.cooldown:
            self._log.append((gesture, action, time.strftime("%H:%M:%S")))
            self._last_action = action
            self._last_time   = now

    def entries(self) -> list:
        return list(self._log)


# ──────────────────────────────────────────────────────────────
#  Drawing Helpers
# ──────────────────────────────────────────────────────────────
FONT    = cv2.FONT_HERSHEY_DUPLEX
C_ACCENT = (0,  220, 180)
C_WHITE  = (240, 240, 240)
C_YELLOW = (30,  210, 255)
C_DIM    = (120, 120, 140)
C_PANEL  = (18,  20,  32)


def blend_rect(frame, pt1, pt2, color, alpha=0.65):
    x1, y1 = pt1; x2, y2 = pt2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub  = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def draw_skeleton(frame, lm, h, w):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 160, 120), 2, cv2.LINE_AA)
    tip_ids = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST}
    for i, (x, y) in enumerate(pts):
        r   = 6 if i in tip_ids else 4
        col = (0, 255, 200) if i in tip_ids else (0, 200, 160)
        cv2.circle(frame, (x, y), r, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (0, 0, 0), 1,  cv2.LINE_AA)


def draw_hud(frame, gesture, action, fps, log, hand_count):
    h, w = frame.shape[:2]

    # Bottom panel
    blend_rect(frame, (0, h - 185), (w, h), C_PANEL, alpha=0.78)
    cv2.line(frame, (0, h - 185), (w, h - 185), C_ACCENT, 1)

    # Title (top-left)
    blend_rect(frame, (8, 8), (320, 52), C_PANEL, alpha=0.82)
    cv2.putText(frame, "Hand Gesture Recognition",
                (18, 37), FONT, 0.58, C_ACCENT, 1, cv2.LINE_AA)

    # FPS / hand count (top-right)
    blend_rect(frame, (w - 215, 8), (w - 8, 52), C_PANEL, alpha=0.82)
    cv2.putText(frame, f"FPS {fps:4.1f}   Hands: {hand_count}",
                (w - 205, 36), FONT, 0.48, C_DIM, 1, cv2.LINE_AA)

    mid = w // 2

    # Current gesture box
    blend_rect(frame, (10, h - 178), (mid - 10, h - 100), (0, 30, 25), alpha=0.6)
    cv2.putText(frame, "GESTURE",
                (22, h - 157), FONT, 0.42, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, gesture,
                (22, h - 118), FONT, 0.88, C_WHITE, 2, cv2.LINE_AA)

    # Action box
    blend_rect(frame, (mid + 10, h - 178), (w - 10, h - 100), (0, 55, 45), alpha=0.6)
    cv2.putText(frame, "ACTION",
                (mid + 22, h - 157), FONT, 0.42, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, action,
                (mid + 22, h - 118), FONT, 0.82, C_ACCENT, 2, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (0, h - 95), (w, h - 95), (40, 45, 60), 1)

    # Recent actions log
    cv2.putText(frame, "Recent:", (14, h - 78), FONT, 0.40, C_DIM, 1, cv2.LINE_AA)
    for i, (g, a, t) in enumerate(list(reversed(log.entries()))[:4]):
        fade = max(0.3, 1.0 - i * 0.2)
        col  = tuple(int(c * fade) for c in C_YELLOW)
        cv2.putText(frame, f"[{t}]  {g}  →  {a}",
                    (14, h - 58 + i * (-18)), FONT, 0.38, col, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "Q / ESC: Quit    S: Screenshot",
                (w - 315, h - 10), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
#  Model Management
# ──────────────────────────────────────────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_NAME = "hand_landmarker.task"


def ensure_model(path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        print(f"✅  Model found: {path}")
        return path
    print(f"⬇  Downloading hand landmark model → {path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        size_kb = os.path.getsize(path) // 1024
        print(f"✅  Downloaded ({size_kb} KB)")
    except Exception as e:
        print(f"❌  Download failed: {e}")
        print(f"\n    Please manually download:\n    {MODEL_URL}\n")
        sys.exit(1)
    return path


# ──────────────────────────────────────────────────────────────
#  Main Loop
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition Demo")
    parser.add_argument("--model",  default=MODEL_NAME,
                        help=f"Path to hand_landmarker.task (default: {MODEL_NAME})")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    model_path = ensure_model(args.model)

    # Build HandLandmarker (VIDEO mode = synchronous per-frame inference)
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.50,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌  Cannot open webcam (index {args.camera}).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    action_log  = ActionLog()
    gesture_buf = deque(maxlen=9)
    prev_ts     = time.time()

    print("\n  🖐  Running – point your hand at the webcam!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)              # mirror for natural selfie view
        h, w  = frame.shape[:2]

        # Inference
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        # FPS
        now, prev_ts = time.time(), time.time()
        fps = 1.0 / max(now - prev_ts, 1e-6)

        gesture    = "—"
        action     = "—"
        hand_count = 0

        if result.hand_landmarks:
            hand_count = len(result.hand_landmarks)
            for idx, lm_list in enumerate(result.hand_landmarks):
                try:
                    hand_side = result.handedness[idx][0].display_name
                except Exception:
                    hand_side = "Right"

                draw_skeleton(frame, lm_list, h, w)
                g, a = classify_gesture(lm_list, hand_side)
                gesture, action = g, a

            # Temporal smoothing: majority vote over last N frames
            gesture_buf.append(gesture)
            stable = max(set(gesture_buf), key=list(gesture_buf).count)
            if stable == gesture:
                action_log.push(gesture, action)

        draw_hud(frame, gesture, action, fps, action_log, hand_count)
        cv2.imshow("Hand Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, frame)
            print(f"📸  Saved: {fname}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  👋  Session ended. Goodbye!\n")


if __name__ == "__main__":
    main()
