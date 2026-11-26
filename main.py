"""
AI Virtual Mouse - Main Application
Integrates real-time MediaPipe Hand Tracking, Gesture Recognition Engine,
Exponential Moving Average (EMA) cursor smoothing, active interaction zone mapping,
debouncing with state hysteresis, and on-screen HUD telemetry.
"""

import os
import time
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller

import config
import util

# Initialize OS Mouse Controllers
pyautogui.FAILSAFE = config.FAILSAFE_ENABLED
pyautogui.PAUSE = config.PYAUTOGUI_PAUSE
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Ensure screenshots directory exists
os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

# Initialize MediaPipe Hands pipeline
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


class VirtualMouseEngine:
    """
    Stateful Gesture Recognition and Cursor Controller.
    Manages coordinate mapping, EMA smoothing, hysteresis debouncing,
    and gesture priority resolution.
    """

    def __init__(self, screen_w=screen_width, screen_h=screen_height):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # Cursor smoothing state
        self.cursor_x = float(screen_w // 2)
        self.cursor_y = float(screen_h // 2)
        self.prev_x = int(self.cursor_x)
        self.prev_y = int(self.cursor_y)

        # Left / Double click pinch state
        self.left_pinch_active = False
        self.last_left_release_time = 0.0
        self.last_left_click_time = 0.0

        # Right click pinch state
        self.right_pinch_active = False
        self.last_right_click_time = 0.0

        # Closed fist & Screenshot state
        self.fist_start_time = None
        self.screenshot_armed = True
        self.last_screenshot_file = ""
        self.screenshot_banner_until = 0.0

        # Telemetry & HUD state
        self.current_gesture = "IDLE"
        self.pinch_left_dist = 0.0
        self.pinch_right_dist = 0.0
        self.fist_progress = 0.0

    def process_hand(self, landmark_list, now=None):
        """
        Evaluates 21 hand landmarks, executes mouse actions based on gesture priority,
        and updates internal state machines.
        """
        if now is None:
            now = time.time()

        if len(landmark_list) < 21:
            self._handle_no_hand()
            return

        # Calculate primary pinch distances (Thumb tip 4 to Index tip 8 / Middle tip 12)
        self.pinch_left_dist = util.get_distance([landmark_list[util.THUMB_TIP], landmark_list[util.INDEX_TIP]])
        self.pinch_right_dist = util.get_distance([landmark_list[util.THUMB_TIP], landmark_list[util.MIDDLE_TIP]])

        # ----------------------------------------------------------------------
        # PRIORITY 1: CLOSED FIST (HOLD FOR 1.0s -> SCREENSHOT)
        # ----------------------------------------------------------------------
        if util.is_fist(landmark_list):
            if self.fist_start_time is None:
                self.fist_start_time = now

            elapsed = now - self.fist_start_time
            self.fist_progress = min(1.0, elapsed / config.FIST_HOLD_DURATION)

            if elapsed >= config.FIST_HOLD_DURATION and self.screenshot_armed:
                self._capture_screenshot(now)
            elif self.screenshot_armed:
                self.current_gesture = f"FIST HOLD ({elapsed:.1f}s / {config.FIST_HOLD_DURATION:.1f}s)"
            else:
                self.current_gesture = "RELEASE FIST TO REARM"
            return
        else:
            # Fist released: rearm screenshot trigger
            self.fist_start_time = None
            self.fist_progress = 0.0
            self.screenshot_armed = True

        # ----------------------------------------------------------------------
        # PRIORITY 2: THUMB + MIDDLE PINCH (RIGHT CLICK)
        # ----------------------------------------------------------------------
        if not self.right_pinch_active and self.pinch_right_dist <= config.PINCH_RIGHT_ACTIVATE:
            # Activation edge
            if now - self.last_right_click_time >= config.CLICK_COOLDOWN:
                mouse.press(Button.right)
                mouse.release(Button.right)
                self.last_right_click_time = now
                self.current_gesture = "RIGHT CLICK"
            self.right_pinch_active = True
            return
        elif self.right_pinch_active:
            if self.pinch_right_dist >= config.PINCH_RIGHT_RELEASE:
                # Release edge (hysteresis)
                self.right_pinch_active = False
            else:
                self.current_gesture = "RIGHT PINCH HELD"
                return

        # ----------------------------------------------------------------------
        # PRIORITY 3: THUMB + INDEX PINCH (LEFT CLICK & DOUBLE CLICK)
        # ----------------------------------------------------------------------
        if not self.left_pinch_active and self.pinch_left_dist <= config.PINCH_LEFT_ACTIVATE:
            # Activation edge: evaluate single vs double click
            time_since_release = now - self.last_left_release_time
            if (time_since_release <= config.DOUBLE_CLICK_MAX_GAP and
                    (now - self.last_left_click_time > 0.05)):
                # Two distinct pinches within interval -> Double click
                pyautogui.doubleClick()
                self.last_left_click_time = now
                self.last_left_release_time = 0.0  # Reset release window
                self.current_gesture = "DOUBLE CLICK"
            else:
                # Single left click
                mouse.press(Button.left)
                mouse.release(Button.left)
                self.last_left_click_time = now
                self.current_gesture = "LEFT CLICK"

            self.left_pinch_active = True
            return
        elif self.left_pinch_active:
            if self.pinch_left_dist >= config.PINCH_LEFT_RELEASE:
                # Release edge (hysteresis)
                self.left_pinch_active = False
                self.last_left_release_time = now
            else:
                self.current_gesture = "LEFT PINCH HELD"
                return

        # ----------------------------------------------------------------------
        # PRIORITY 4: INDEX EXTENDED (MOVE CURSOR)
        # ----------------------------------------------------------------------
        index_extended = util.is_finger_extended(
            landmark_list, util.INDEX_MCP, util.INDEX_PIP, util.INDEX_TIP, config.FINGER_EXTENDED_ANGLE
        )

        if index_extended:
            index_norm_x, index_norm_y = landmark_list[util.INDEX_TIP]
            target_x, target_y = util.map_coordinates_to_screen(
                index_norm_x, index_norm_y, self.screen_w, self.screen_h,
                config.ZONE_LEFT, config.ZONE_RIGHT, config.ZONE_TOP, config.ZONE_BOTTOM
            )

            # Apply Exponential Moving Average (EMA) smoothing
            self.cursor_x = config.SMOOTHING_ALPHA * target_x + (1.0 - config.SMOOTHING_ALPHA) * self.cursor_x
            self.cursor_y = config.SMOOTHING_ALPHA * target_y + (1.0 - config.SMOOTHING_ALPHA) * self.cursor_y

            # Deadzone check to suppress micro-jitter
            dx = self.cursor_x - self.prev_x
            dy = self.cursor_y - self.prev_y
            if np.hypot(dx, dy) >= config.DEADZONE_PIXELS:
                px = int(np.clip(self.cursor_x, 0, self.screen_w - 1))
                py = int(np.clip(self.cursor_y, 0, self.screen_h - 1))
                pyautogui.moveTo(px, py)
                self.prev_x = px
                self.prev_y = py

            self.current_gesture = "MOVE CURSOR"
            return

        # ----------------------------------------------------------------------
        # PRIORITY 5: IDLE POSE
        # ----------------------------------------------------------------------
        self.current_gesture = "IDLE"

    def _capture_screenshot(self, now):
        """Captures a single screenshot, saves it to disk, and updates notification state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(config.SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
        try:
            img = pyautogui.screenshot()
            img.save(filename)
            self.last_screenshot_file = os.path.basename(filename)
            self.screenshot_banner_until = now + config.SCREENSHOT_BANNER_DURATION
            self.screenshot_armed = False
            self.current_gesture = "SCREENSHOT CAPTURED"
        except Exception as e:
            print(f"[ERROR] Failed to save screenshot: {e}")

    def _handle_no_hand(self):
        """Resets dynamic tracking states when hand is lost."""
        self.fist_start_time = None
        self.fist_progress = 0.0
        self.left_pinch_active = False
        self.right_pinch_active = False
        self.current_gesture = "SEARCHING HAND"


def draw_hud(frame, fps, engine, hand_detected):
    """
    Renders professional telemetry overlay onto the OpenCV display frame.
    """
    h, w, _ = frame.shape
    now = time.time()

    # 1. Draw Active Interaction Zone
    zx1 = int(config.ZONE_LEFT * w)
    zx2 = int(config.ZONE_RIGHT * w)
    zy1 = int(config.ZONE_TOP * h)
    zy2 = int(config.ZONE_BOTTOM * h)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 180, 0), 2)
    cv2.putText(frame, "Active Tracking Zone", (zx1 + 8, zy1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

    # 2. Draw Top Status Header Banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Telemetry Left: FPS & Tracking Status
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "Hand: TRACKING" if hand_detected else "Hand: SEARCHING"
    cv2.putText(frame, status_text, (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    # Telemetry Center: Active Gesture
    gesture_color = (0, 255, 255) if hand_detected else (160, 160, 160)
    cv2.putText(frame, f"Gesture: {engine.current_gesture}", (180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, gesture_color, 2)

    # Telemetry Metrics: Pinch Distances
    pinch_info = f"Pinch L: {engine.pinch_left_dist:.1f} | Pinch R: {engine.pinch_right_dist:.1f}"
    cv2.putText(frame, pinch_info, (180, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 3. Fist Hold Progress Bar (Screenshot countdown)
    if engine.fist_start_time is not None and engine.screenshot_armed:
        bar_x1, bar_y1 = int(w * 0.25), 85
        bar_x2, bar_y2 = int(w * 0.75), 105
        bar_w = bar_x2 - bar_x1
        fill_w = int(bar_w * engine.fist_progress)

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 140, 255), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)
        cv2.putText(frame, "Capturing Screenshot...", (bar_x1 + 10, bar_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

    # 4. Screenshot Confirmation Banner
    if now < engine.screenshot_banner_until:
        banner_y1, banner_y2 = h - 60, h - 15
        cv2.rectangle(frame, (20, banner_y1), (w - 20, banner_y2), (0, 180, 0), -1)
        cv2.putText(frame, f"[SUCCESS] Saved: {engine.last_screenshot_file}", (35, banner_y2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def open_camera():
    """Attempts to initialize camera on index 0, falling back to 1 or 2 if unavailable."""
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Set preferred resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[INFO] Camera initialized successfully on index {idx}")
                return cap
            cap.release()
    return None


def main():
    print("[INFO] Starting AI Virtual Mouse...")
    print("[INFO] Controls:")
    print("       - Point Index: Move Cursor")
    print("       - Thumb + Index Pinch: Left Click")
    print("       - Quick Double Pinch: Double Click")
    print("       - Thumb + Middle Pinch: Right Click")
    print("       - Closed Fist (Hold 1.0s): Take Screenshot")
    print("       - Press 'q' on camera window to Exit")

    cap = open_camera()
    if cap is None:
        print("[ERROR] Could not open camera on any index (0, 1, 2).")
        print("[INFO] Please verify webcam connection and Windows privacy permissions.")
        return

    engine = VirtualMouseEngine(screen_width, screen_height)
    prev_time = time.time()
    fps = 30.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame capture failed or camera disconnected.")
                break

            # Mirror frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frame_rgb)

            landmark_list = []
            hand_detected = False

            if processed.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = processed.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            # Execute gesture engine
            now = time.time()
            engine.process_hand(landmark_list, now=now)

            # Calculate FPS with rolling smoothing
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps

            # Render visual HUD overlay
            draw_hud(frame, fps, engine, hand_detected)

            cv2.imshow("AI Virtual Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User initiated shutdown ('q' pressed).")
                break

    except Exception as e:
        print(f"[EXCEPTION] Runtime error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and application cleanly terminated.")


if __name__ == "__main__":
    main()
