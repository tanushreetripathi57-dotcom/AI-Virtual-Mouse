# 🖐️ AI Virtual Mouse using Computer Vision

An advanced, real-time AI-powered Virtual Mouse system built using **OpenCV**, **MediaPipe Hands**, **NumPy**, **PyAutoGUI**, and **pynput**. Control your computer's mouse cursor, click buttons, execute double clicks, and capture screenshots using natural hand gestures through a standard webcam — without touching physical hardware.

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

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/NaveenSalve/virtual-mouse.git
cd virtual-mouse
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Virtual Mouse
```bash
python main.py
```

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

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Computer Vision:** OpenCV (`cv2`)
* **ML / Perception:** Google MediaPipe Hands
* **Math & Geometry:** NumPy
* **OS Automation:** PyAutoGUI & pynput

---

## 👤 Author & Acknowledgments

* **Developer:** [Naveen Salve](https://github.com/NaveenSalve)
* Developed as an advanced Computer Vision & Human-Computer Interaction (HCI) project.
