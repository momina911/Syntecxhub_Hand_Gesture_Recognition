# Syntecxhub_Hand_Gesture_Recognition
A real-time hand gesture recognition system using MediaPipe and OpenCV that detects hand landmarks via webcam, classifies gestures, and maps them to media control actions.

## Project Overview
This project uses MediaPipe's Hand Landmarker model to detect 21 3D landmarks per hand in real time. A custom finger-state classifier analyzes joint positions to recognize 8 distinct gestures, which are mapped to media player actions like play, pause, and volume control.

## Features

Real-time hand landmark detection via webcam
Supports up to 2 hands simultaneously
Classifies 8 gestures with high accuracy
Temporal smoothing using a 9-frame majority-vote buffer
Action cooldown logger to prevent event flooding
On-screen HUD overlay showing gesture, action, FPS, and action history
Screenshot capture with a single keypress
Auto-downloads the MediaPipe hand landmark model on first run


## Gestures and Actions
GestureHand ShapeMapped Action✋ Open PalmAll fingers extended▶ Play✊ FistAll fingers closed⏸ Pause👍 Thumbs UpThumb up, others closed🔊 Volume Up👎 Thumbs DownThumb down, others closed🔉 Volume Down☝️ One FingerIndex only extended⏭ Next Track✌️ Two FingersIndex + middle extended⏮ Previous Track🤟 Three FingersIndex + middle + ring🔀 Shuffle🤙 PinkyPinky only extended🔇 Mute

## Tech Stack

Python 3.10+
MediaPipe — Hand landmark detection (Tasks API v0.10+)
OpenCV — Webcam capture and HUD rendering
NumPy — Frame processing and overlay blending

## Setup and Installation
1. Clone the repository
bashgit clone https://github.com/momina911/Syntecxhub_Hand_Gesture_Recognition
cd hand-gesture-recognition
2. Create and activate a virtual environment
bash# Windows
python -m venv venv
venv\Scripts\activate

## macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bashpip install mediapipe opencv-python numpy
4. Run the demo
bashpython gesture_control.py
The hand landmark model (~8 MB) will be downloaded automatically on the first run.

## Controls
KeyActionQ or ESCQuit the applicationSSave a screenshot

## How It Works

Capture — OpenCV reads frames from the webcam and flips them for a natural mirror view
Detect — MediaPipe processes each frame and returns 21 normalized (x, y, z) landmarks per hand
Classify — The finger state classifier compares tip vs. PIP joint positions to determine which fingers are extended
Smooth — A 9-frame majority-vote buffer filters out flickering between gestures
Dispatch — The stable gesture is matched to an action and logged with a 1.5-second cooldown
Display — OpenCV renders the skeleton overlay and HUD panel on the live frame
