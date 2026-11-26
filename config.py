"""
AI Virtual Mouse - Configuration Parameters
Centralized settings for gestures, smoothing, interaction zone, and timings.
"""

# ==============================================================================
# 1. SCREEN & INTERACTION ZONE (CAMERA TO SCREEN MAPPING)
# ==============================================================================
# Defines the normalized sub-rectangle inside the camera frame [0.0 to 1.0].
# Only hand movements inside this box are mapped to the full screen.
ZONE_LEFT = 0.20        # Left margin (20% of frame width)
ZONE_RIGHT = 0.80       # Right margin (80% of frame width)
ZONE_TOP = 0.15         # Top margin (15% of frame height)
ZONE_BOTTOM = 0.70      # Bottom margin (70% of frame height)

# ==============================================================================
# 2. CURSOR SMOOTHING & SENSITIVITY
# ==============================================================================
# Exponential Moving Average (EMA) weight: new_pos = alpha * target + (1 - alpha) * prev_pos
# Lower alpha = smoother but slight lag; Higher alpha = faster but more jitter.
# 0.30 - 0.35 gives an ideal balance between responsiveness and jitter elimination.
SMOOTHING_ALPHA = 0.35

# Deadzone in pixels: if mouse moves less than this distance, ignore micro-tremors.
DEADZONE_PIXELS = 4

# Cursor movement sensitivity multiplier
SENSITIVITY = 1.0

# ==============================================================================
# 3. GESTURE DETECTION THRESHOLDS (HYSTERESIS & GEOMETRY)
# ==============================================================================
# Distance scale is 0 to 1000 (0.0 to 1.0 in normalized camera coordinates).

# Thumb-Index pinch (Left Click / Double Click):
PINCH_LEFT_ACTIVATE = 38.0     # Distance <= 38 activates pinch
PINCH_LEFT_RELEASE = 58.0      # Distance >= 58 releases pinch (hysteresis)

# Thumb-Middle pinch (Right Click):
PINCH_RIGHT_ACTIVATE = 38.0    # Distance <= 38 activates pinch
PINCH_RIGHT_RELEASE = 58.0     # Distance >= 58 releases pinch (hysteresis)

# Finger angles (MCP -> PIP -> TIP) in degrees:
FINGER_EXTENDED_ANGLE = 125.0  # Angle > 125° considered extended/straight
FINGER_CURLED_ANGLE = 85.0     # Angle < 85° considered curled/bent

# ==============================================================================
# 4. TIMING & DEBOUNCING (SECONDS)
# ==============================================================================
# Maximum time gap between two distinct pinches to register a Double Click
DOUBLE_CLICK_MAX_GAP = 0.40

# Minimum cooldown between distinct clicks to avoid double-firing
CLICK_COOLDOWN = 0.15

# Continuous hold duration required to trigger a Screenshot (Closed Fist)
FIST_HOLD_DURATION = 1.0

# Duration to display the screenshot notification banner on screen
SCREENSHOT_BANNER_DURATION = 2.0

# ==============================================================================
# 5. STORAGE & SAFETY
# ==============================================================================
SCREENSHOT_DIR = "screenshots"
FAILSAFE_ENABLED = True
PYAUTOGUI_PAUSE = 0.0
