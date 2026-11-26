# AI Virtual Mouse

This project turns a webcam into a basic hand-controlled mouse. OpenCV captures the video, MediaPipe detects hand landmarks, and Python translates the gestures into cursor movement and mouse actions.

## Features

- Real-time index-finger cursor control
- Left click and double-click gestures
- Right click gesture
- Screenshot capture with a closed-fist hold
- Cursor smoothing and reduced pointer jitter
- On-screen FPS, gesture, pinch, and tracking information
- Automatic camera fallback across camera indexes 0, 1, and 2

---

## 🌟 Key Features

* **Real-Time 21 Hand Landmark Tracking:** Powered by Google MediaPipe's lightweight dual neural network architecture (BlazePalm Detector SSD + 3D Landmark Regression Network).
* **Smooth & Tremor-Free Cursor Control:** Integrated **Exponential Moving Average (EMA)** smoothing ($\alpha = 0.35$) with sub-pixel **Deadzone Jitter Filtering** ($4\text{ px}$) to eliminate hand shaking.
* **Active Interaction Workspace:** Normalized camera-to-screen coordinate mapping with an on-screen bounded interaction box ($20\%$–$80\%$ horizontal, $15\%$–$70\%$ vertical) allowing full-screen reach with minimal arm strain.
* **Hysteresis Debounced Clicking:** Prevents rapid-fire click flooding using dual activation ($\le 38.0$) and release ($\ge 58.0$) Euclidean distance thresholds.
* **Natural Double-Click Detection:** Evaluates two sequential quick left pinches within $0.40\text{ seconds}$ to execute native OS double clicks.
* **Closed Fist Screenshot Capture:** Holding a closed fist continuously for $1.0\text{ second}$ captures and saves a high-resolution screenshot to the `screenshots/` directory, complete with countdown HUD progress and single-shot latching.
* **Rich On-Screen HUD Telemetry:** Displays smoothed live FPS, tracking confidence state, active gesture banner, real-time pinch metrics, and active tracking zone borders.
* **Multi-Camera Hardware Fallback:** Automatically probes and connects across camera indices `0`, `1`, and `2`.

---

## 🎮 Gesture Controls Reference

| Hand Gesture | Action Triggered | Technical Condition |
|---|---|---|
| ☝️ **Index Finger Extended** | **Move Mouse Cursor** | Index PIP angle $\ge 125^\circ$, Middle & Thumb open |
| 🤏 **Thumb + Index Pinch** | **Left Mouse Click** | Euclidean dist(Landmark 4, Landmark 8) $\le 38.0$ |
| ⚡ **Quick Double Pinch** | **Double Click** | 2 consecutive left pinches within $0.40\text{ seconds}$ |
| ✌️ **Thumb + Middle Pinch** | **Right Mouse Click** | Euclidean dist(Landmark 4, Landmark 12) $\le 38.0$ |
| ✊ **Closed Fist (Hold 1.0s)** | **Capture Screenshot** | All 4 fingers curled $\le 95^\circ$, Thumb tucked inward |
| 🛑 **Press 'q' Key** | **Quit Application** | Graceful camera release and window teardown |

---

## 🏗️ Project Architecture

```
VirtualMouse/
├── config.py             # Centralized settings (margins, EMA alpha, thresholds, timings)
├── util.py               # Geometric math, vector angles, distances, finger state logic
├── main.py               # Core orchestrator, state machine, camera loop & HUD overlay
├── test_virtual_mouse.py # Comprehensive 10-suite automated test verification
├── requirements.txt      # Dependency specification
└── screenshots/          # Auto-created directory for captured screenshots
```

---

## Installation and running

Python 3.12 is recommended for this setup. Create a virtual environment in the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the application:

```powershell
.\.venv\Scripts\python.exe main.py
```

Press `q` in the camera window to exit.

---

## 🧪 Automated Testing

A complete automated unit and integration test suite is included to verify all geometric, temporal, and state machine logic:

```bash
python test_virtual_mouse.py
```

### Test Coverage:
* Interior angle calculation with $180^\circ$ boundary wrap normalization.
* Euclidean distance mapping ($0$–$1000$ scale).
* Finger state classification (extended vs. curled vs. closed fist).
* Interaction zone boundary mapping and screen clamping.
* EMA cursor smoothing and deadzone jitter suppression.
* Left click hysteresis and single-trigger latching.
* Quick double-pinch detection within timing gap.
* Right click hysteresis and release rearming.
* 1-second continuous closed fist screenshot hold and single-shot latch.
* Strict gesture priority resolution (fist takes precedence over pinch).

---

## Technology used

- Python 3.12
- OpenCV
- MediaPipe Hands
- NumPy
- PyAutoGUI
- pynput

## Troubleshooting

- If the webcam preview is black, close other apps using the camera and restart the virtual mouse.
- Windows camera access must be enabled under **Privacy & security > Camera**.
- The app uses DirectShow for the webcam on Windows.
- If `mediapipe` does not provide `mp.solutions.hands`, use Python 3.12 and `mediapipe==0.10.14`.

## Author

Tanushree Tripathi
