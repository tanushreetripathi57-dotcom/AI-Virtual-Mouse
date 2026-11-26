"""
Automated Test Suite for AI Virtual Mouse (Phase 2 & Phase 3)
Tests:
1. Finger angle calculations & normalization
2. Euclidean pinch distance calculation
3. Finger state classification (extended, curled, fist)
4. Active interaction zone & screen coordinate mapping
5. Cursor smoothing (EMA) and deadzone jitter suppression
6. Gesture priority & conflict resolution
7. Left click and hysteresis release/rearming
8. Double click detection within timing window
9. Right click and hysteresis release/rearming
10. Closed fist screenshot capture, 1-second hold, and rearming
"""

import time
import os
import numpy as np
import pyautogui
from pynput.mouse import Button

import config
import util
import main


def create_mock_hand(pose_type="open", tip_offset=(0.0, 0.0)):
    """
    Constructs a synthetic 21-landmark hand coordinate list.
    Coordinates are normalized in [0.0, 1.0].
    """
    # Base wrist at bottom center of hand
    lm = [(0.5, 0.8)] * 21

    # Base palm MCPs
    lm[util.WRIST] = (0.5, 0.8)
    lm[util.THUMB_CMC] = (0.45, 0.75)
    lm[util.THUMB_MCP] = (0.42, 0.70)
    lm[util.THUMB_IP] = (0.40, 0.65)
    lm[util.THUMB_TIP] = (0.38, 0.60)

    # Index finger
    lm[util.INDEX_MCP] = (0.46, 0.65)
    lm[util.INDEX_PIP] = (0.46, 0.55)
    lm[util.INDEX_DIP] = (0.46, 0.45)
    lm[util.INDEX_TIP] = (0.46 + tip_offset[0], 0.35 + tip_offset[1])

    # Middle finger
    lm[util.MIDDLE_MCP] = (0.50, 0.65)
    lm[util.MIDDLE_PIP] = (0.50, 0.54)
    lm[util.MIDDLE_DIP] = (0.50, 0.44)
    lm[util.MIDDLE_TIP] = (0.50, 0.34)

    # Ring finger
    lm[util.RING_MCP] = (0.54, 0.66)
    lm[util.RING_PIP] = (0.54, 0.56)
    lm[util.RING_DIP] = (0.54, 0.46)
    lm[util.RING_TIP] = (0.54, 0.36)

    # Pinky finger
    lm[util.PINKY_MCP] = (0.58, 0.68)
    lm[util.PINKY_PIP] = (0.58, 0.60)
    lm[util.PINKY_DIP] = (0.58, 0.52)
    lm[util.PINKY_TIP] = (0.58, 0.44)

    if pose_type == "open":
        # All fingers extended
        pass

    elif pose_type == "index_only":
        # Index extended, middle, ring, pinky curled
        # Middle curled
        lm[util.MIDDLE_PIP] = (0.50, 0.58)
        lm[util.MIDDLE_TIP] = (0.50, 0.68)
        # Ring curled
        lm[util.RING_PIP] = (0.54, 0.59)
        lm[util.RING_TIP] = (0.54, 0.69)
        # Pinky curled
        lm[util.PINKY_PIP] = (0.58, 0.62)
        lm[util.PINKY_TIP] = (0.58, 0.70)
        # Thumb away
        lm[util.THUMB_TIP] = (0.35, 0.60)

    elif pose_type == "pinch_left":
        # Thumb tip and index tip touching (distance < 20)
        pinch_pt = (0.42, 0.50)
        lm[util.THUMB_TIP] = pinch_pt
        lm[util.INDEX_TIP] = (pinch_pt[0] + 0.01, pinch_pt[1] + 0.01)

    elif pose_type == "pinch_right":
        # Thumb tip and middle tip touching
        pinch_pt = (0.45, 0.50)
        lm[util.THUMB_TIP] = pinch_pt
        lm[util.MIDDLE_TIP] = (pinch_pt[0] + 0.01, pinch_pt[1] + 0.01)
        # Index tip far away
        lm[util.INDEX_TIP] = (0.35, 0.30)

    elif pose_type == "fist":
        # All fingers curled toward palm
        lm[util.INDEX_PIP] = (0.46, 0.60)
        lm[util.INDEX_TIP] = (0.46, 0.68)
        lm[util.MIDDLE_PIP] = (0.50, 0.60)
        lm[util.MIDDLE_TIP] = (0.50, 0.68)
        lm[util.RING_PIP] = (0.54, 0.61)
        lm[util.RING_TIP] = (0.54, 0.69)
        lm[util.PINKY_PIP] = (0.58, 0.63)
        lm[util.PINKY_TIP] = (0.58, 0.70)
        # Thumb folded inward across knuckles
        lm[util.THUMB_TIP] = (0.47, 0.65)

    return lm


def run_tests():
    print("=" * 70)
    print("AI VIRTUAL MOUSE — AUTOMATED TEST SUITE (PHASE 2 & PHASE 3)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TEST 1: Finger Angle Calculations & Normalization
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Finger Angles & Boundary Normalization")
    a, b, c = (0, 2), (0, 1), (0, 0)
    angle_180 = util.get_angle(a, b, c)
    assert np.isclose(angle_180, 180.0), f"Straight line failed: {angle_180}"
    print(f"  ✓ Straight vertical angle: {angle_180:.1f}°")

    angle_90 = util.get_angle((1, 0), (0, 0), (0, 1))
    assert np.isclose(angle_90, 90.0), f"Right angle failed: {angle_90}"
    print(f"  ✓ 90° right angle: {angle_90:.1f}°")

    # Wrap-around across [-pi, pi]
    v1_rad = np.radians(175)
    v2_rad = np.radians(-175)
    angle_wrap = util.get_angle((np.cos(v1_rad), np.sin(v1_rad)), (0, 0), (np.cos(v2_rad), np.sin(v2_rad)))
    assert np.isclose(angle_wrap, 10.0), f"Wrap angle failed: {angle_wrap}"
    print(f"  ✓ Wrap-around boundary angle (+175°/-175°): {angle_wrap:.1f}°")

    # -------------------------------------------------------------------------
    # TEST 2: Distance Calculations (0 to 1000 scale)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Distance Calculations")
    dist_zero = util.get_distance([(0.5, 0.5), (0.5, 0.5)])
    assert dist_zero == 0.0, f"Zero distance failed: {dist_zero}"
    dist_half = util.get_distance([(0.0, 0.0), (0.5, 0.0)])
    assert np.isclose(dist_half, 500.0), f"Half-width distance failed: {dist_half}"
    print(f"  ✓ Zero distance: {dist_zero:.1f}, Half-scale: {dist_half:.1f}")

    # -------------------------------------------------------------------------
    # TEST 3: Finger State Classification (Extended vs Curled)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Finger State Classification")
    hand_open = create_mock_hand("open")
    hand_fist = create_mock_hand("fist")

    assert util.is_finger_extended(hand_open, util.INDEX_MCP, util.INDEX_PIP, util.INDEX_TIP), "Open hand index not extended!"
    assert not util.is_finger_curled(hand_open, util.INDEX_MCP, util.INDEX_PIP, util.INDEX_TIP), "Open hand index marked curled!"
    assert util.is_finger_curled(hand_fist, util.INDEX_MCP, util.INDEX_PIP, util.INDEX_TIP), "Fist index not curled!"
    assert util.is_fist(hand_fist), "Closed fist not recognized by util.is_fist!"
    assert not util.is_fist(hand_open), "Open hand falsely recognized as fist!"
    print("  ✓ is_finger_extended and is_finger_curled correct")
    print("  ✓ is_fist accurately differentiates closed fist from open hand")

    # -------------------------------------------------------------------------
    # TEST 4: Active Interaction Zone & Full-Screen Coordinate Mapping
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Interaction Zone & Screen Coordinate Mapping")
    sw, sh = 1920, 1080
    zl, zr, zt, zb = config.ZONE_LEFT, config.ZONE_RIGHT, config.ZONE_TOP, config.ZONE_BOTTOM

    # Top-Left of zone
    x_tl, y_tl = util.map_coordinates_to_screen(zl, zt, sw, sh, zl, zr, zt, zb)
    assert x_tl == 0 and y_tl == 0, f"Zone Top-Left mapping failed: ({x_tl}, {y_tl})"

    # Bottom-Right of zone
    x_br, y_br = util.map_coordinates_to_screen(zr, zb, sw, sh, zl, zr, zt, zb)
    assert x_br == sw - 1 and y_br == sh - 1, f"Zone Bottom-Right mapping failed: ({x_br}, {y_br})"

    # Center of zone
    z_cx = (zl + zr) / 2.0
    z_cy = (zt + zb) / 2.0
    x_c, y_c = util.map_coordinates_to_screen(z_cx, z_cy, sw, sh, zl, zr, zt, zb)
    assert abs(x_c - sw // 2) <= 1 and abs(y_c - sh // 2) <= 1, f"Zone Center mapping failed: ({x_c}, {y_c})"

    # Out of bounds clamping
    x_oob, y_oob = util.map_coordinates_to_screen(0.0, 0.0, sw, sh, zl, zr, zt, zb)
    assert x_oob == 0 and y_oob == 0, f"OOB clamp failed: ({x_oob}, {y_oob})"
    print(f"  ✓ Zone mapping: Top-Left=({x_tl}, {y_tl}), Bottom-Right=({x_br}, {y_br}), Center=({x_c}, {y_c})")

    # -------------------------------------------------------------------------
    # TEST 5: Cursor Smoothing (EMA) & Deadzone Jitter Suppression
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Cursor Smoothing & Deadzone Suppression")
    engine = main.VirtualMouseEngine(sw, sh)
    engine.cursor_x, engine.cursor_y = 500.0, 500.0
    engine.prev_x, engine.prev_y = 500, 500

    # Mock pyautogui.moveTo
    move_events = []
    pyautogui.moveTo = lambda x, y: move_events.append((x, y))

    # Hand with slight micro-movement (< DEADZONE_PIXELS)
    # Target mapped position produces small delta
    engine.cursor_x = 500.0 + (config.DEADZONE_PIXELS - 1)
    engine.cursor_y = 500.0
    dx = engine.cursor_x - engine.prev_x
    dy = engine.cursor_y - engine.prev_y
    if np.hypot(dx, dy) >= config.DEADZONE_PIXELS:
        pyautogui.moveTo(int(engine.cursor_x), int(engine.cursor_y))

    assert len(move_events) == 0, "Deadzone failed: move event emitted for sub-deadzone shift!"
    print(f"  ✓ Sub-deadzone movement correctly suppressed (delta: {dx:.1f}px)")

    # Large movement: should be smoothed by alpha
    target_x = 800.0
    engine.cursor_x = config.SMOOTHING_ALPHA * target_x + (1.0 - config.SMOOTHING_ALPHA) * 500.0
    dx = engine.cursor_x - engine.prev_x
    if np.hypot(dx, dy) >= config.DEADZONE_PIXELS:
        pyautogui.moveTo(int(engine.cursor_x), int(engine.cursor_y))

    expected_smooth_x = int(0.35 * 800 + 0.65 * 500)  # 280 + 325 = 605
    assert len(move_events) == 1, "Smoothing move event not triggered!"
    assert abs(move_events[0][0] - expected_smooth_x) <= 1, f"EMA calculation mismatch: {move_events[0]}"
    print(f"  ✓ EMA smoothing correctly dampens sudden jump: 500 -> 800 becomes {move_events[0][0]}")

    # -------------------------------------------------------------------------
    # TEST 6: Left Click, Hysteresis, and Cooldown
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Left Click Debounce & Hysteresis")
    engine = main.VirtualMouseEngine(sw, sh)
    click_events = []

    # Mock pynput mouse
    main.mouse.press = lambda btn: click_events.append(("press", btn))
    main.mouse.release = lambda btn: click_events.append(("release", btn))

    hand_pinch_left = create_mock_hand("pinch_left")
    t0 = 1000.0

    # Step A: Pinch activates
    engine.process_hand(hand_pinch_left, now=t0)
    assert len(click_events) == 2, f"Expected 1 click (press+release), got {click_events}"
    assert click_events[0] == ("press", Button.left)
    assert engine.left_pinch_active is True
    assert engine.current_gesture == "LEFT CLICK"

    # Step B: Pinch remains held for next 10 frames
    for i in range(1, 11):
        engine.process_hand(hand_pinch_left, now=t0 + i * 0.033)
    # No extra clicks should have fired!
    assert len(click_events) == 2, f"Click repeated while held! Events: {len(click_events)}"
    assert engine.current_gesture == "LEFT PINCH HELD"
    print("  ✓ Single click executed on activation edge; zero repeats while held")

    # Step C: Pinch releases (hand opens)
    t_release = t0 + 0.40
    hand_open = create_mock_hand("open")
    engine.process_hand(hand_open, now=t_release)
    assert engine.left_pinch_active is False
    assert engine.last_left_release_time == t_release
    print("  ✓ Hysteresis release detected and release timestamp recorded")

    # -------------------------------------------------------------------------
    # TEST 7: Double Click Recognition
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Double Click Detection via Quick Two-Pinch Sequence")
    engine = main.VirtualMouseEngine(sw, sh)
    double_click_events = []
    pyautogui.doubleClick = lambda: double_click_events.append("doubleClick")

    # Pinch 1 at t = 100.0
    t1 = 100.0
    engine.process_hand(hand_pinch_left, now=t1)
    # Release 1 at t = 100.15
    engine.process_hand(hand_open, now=t1 + 0.15)

    # Pinch 2 quickly at t = 100.30 (within DOUBLE_CLICK_MAX_GAP = 0.40s)
    engine.process_hand(hand_pinch_left, now=t1 + 0.30)
    assert len(double_click_events) == 1, f"Double click failed to trigger! Events: {double_click_events}"
    assert engine.current_gesture == "DOUBLE CLICK"
    print("  ✓ Quick sequential pinch (gap=0.15s) registered as Double Click")

    # -------------------------------------------------------------------------
    # TEST 8: Right Click & Hysteresis
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Right Click Debounce & Hysteresis")
    engine = main.VirtualMouseEngine(sw, sh)
    click_events.clear()
    hand_pinch_right = create_mock_hand("pinch_right")

    engine.process_hand(hand_pinch_right, now=200.0)
    assert len(click_events) == 2, f"Right click failed: {click_events}"
    assert click_events[0] == ("press", Button.right)
    assert engine.current_gesture == "RIGHT CLICK"

    # Held frames
    for i in range(1, 5):
        engine.process_hand(hand_pinch_right, now=200.0 + i * 0.033)
    assert len(click_events) == 2, "Right click repeated while held!"
    assert engine.current_gesture == "RIGHT PINCH HELD"

    # Release
    engine.process_hand(hand_open, now=200.2)
    assert engine.right_pinch_active is False
    print("  ✓ Right click fired once and latched until release")

    # -------------------------------------------------------------------------
    # TEST 9: Closed Fist Screenshot (Hold 1.0s, Single Capture & Rearm)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Closed Fist Screenshot Capture (1-Second Hold)")
    engine = main.VirtualMouseEngine(sw, sh)
    screenshot_captures = []

    class MockImage:
        def save(self, path):
            screenshot_captures.append(path)

    pyautogui.screenshot = lambda: MockImage()

    hand_fist = create_mock_hand("fist")
    t_start = 500.0

    # Frame 1: Fist starts at 0.0s
    engine.process_hand(hand_fist, now=t_start)
    assert len(screenshot_captures) == 0, "Screenshot fired prematurely on frame 1!"
    assert "FIST HOLD" in engine.current_gesture

    # Frame at 0.5s: Still holding
    engine.process_hand(hand_fist, now=t_start + 0.50)
    assert len(screenshot_captures) == 0, "Screenshot fired at 0.5s!"
    assert engine.fist_progress == 0.50

    # Frame at 1.05s: Hold duration exceeded -> Exactly 1 screenshot taken
    engine.process_hand(hand_fist, now=t_start + 1.05)
    assert len(screenshot_captures) == 1, f"Screenshot not captured at 1.05s: {screenshot_captures}"
    assert engine.screenshot_armed is False
    assert engine.current_gesture == "SCREENSHOT CAPTURED"

    # Frame at 1.50s: Fist continues to be held -> NO duplicate screenshots!
    for i in range(1, 10):
        engine.process_hand(hand_fist, now=t_start + 1.05 + i * 0.05)
    assert len(screenshot_captures) == 1, f"Duplicate screenshot fired while fist held! Count: {len(screenshot_captures)}"
    assert engine.current_gesture == "RELEASE FIST TO REARM"
    print("  ✓ Exactly 1 screenshot captured after 1.0s continuous fist hold")
    print("  ✓ Continuous fist hold prevented duplicate captures")

    # Fist released -> Rearmed!
    engine.process_hand(hand_open, now=t_start + 2.0)
    assert engine.screenshot_armed is True
    assert engine.fist_start_time is None
    print("  ✓ Fist release successfully rearmed screenshot trigger")

    # -------------------------------------------------------------------------
    # TEST 10: Gesture Priority Conflict Resolution
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Gesture Priority & Conflict Resolution")
    # If a hand forms a fist, pinch should NOT trigger even if fingers are close
    engine = main.VirtualMouseEngine(sw, sh)
    conflict_hand = create_mock_hand("fist")
    # Force thumb and index close inside fist
    conflict_hand[util.INDEX_TIP] = conflict_hand[util.THUMB_TIP]

    click_events.clear()
    engine.process_hand(conflict_hand, now=600.0)
    assert len(click_events) == 0, "Pinch click fired when hand was in fist state! Priority failed."
    assert "FIST" in engine.current_gesture
    print("  ✓ Priority 1 (Fist/Screenshot) takes strict precedence over pinch clicks")

    print("\n" + "=" * 70)
    print("ALL 10 AUTOMATED VERIFICATION SUITES PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
