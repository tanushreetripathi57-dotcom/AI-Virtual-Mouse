import numpy as np

# Landmark Indices for MediaPipe Hands
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def get_angle(a, b, c):
    """
    Calculates the interior angle formed at joint 'b' by vectors ba and bc.
    Returns normalized angle in degrees [0.0, 180.0].
    """
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians)) % 360
    if angle > 180:
        angle = 360 - angle
    return float(angle)


def get_distance(landmark_list):
    """
    Calculates Euclidean distance between two landmarks.
    Maps normalized [0.0, 1.0] to a 0-1000 scale.
    """
    if len(landmark_list) < 2:
        return 0.0
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return float(np.interp(L, [0.0, 1.0], [0.0, 1000.0]))


def is_finger_extended(landmark_list, mcp_idx, pip_idx, tip_idx, threshold=125.0):
    """
    Checks if a finger is extended based on joint angle at PIP.
    """
    angle = get_angle(landmark_list[mcp_idx], landmark_list[pip_idx], landmark_list[tip_idx])
    return angle >= threshold


def is_finger_curled(landmark_list, mcp_idx, pip_idx, tip_idx, threshold=85.0):
    """
    Checks if a finger is curled/bent based on joint angle at PIP.
    """
    angle = get_angle(landmark_list[mcp_idx], landmark_list[pip_idx], landmark_list[tip_idx])
    return angle <= threshold


def is_fist(landmark_list, curl_threshold=95.0):
    """
    Checks if hand forms a closed fist: all 4 fingers (Index, Middle, Ring, Pinky) curled,
    and thumb is folded inward toward index knuckle.
    """
    if len(landmark_list) < 21:
        return False

    index_curled = is_finger_curled(landmark_list, INDEX_MCP, INDEX_PIP, INDEX_TIP, curl_threshold)
    middle_curled = is_finger_curled(landmark_list, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, curl_threshold)
    ring_curled = is_finger_curled(landmark_list, RING_MCP, RING_PIP, RING_TIP, curl_threshold)
    pinky_curled = is_finger_curled(landmark_list, PINKY_MCP, PINKY_PIP, PINKY_TIP, curl_threshold)

    # Thumb is curled if thumb tip (4) is close to index MCP (5) or thumb IP angle is bent
    thumb_dist = get_distance([landmark_list[THUMB_TIP], landmark_list[INDEX_MCP]])
    thumb_folded = thumb_dist < 100.0 or is_finger_curled(landmark_list, THUMB_MCP, THUMB_IP, THUMB_TIP, 140.0)

    return index_curled and middle_curled and ring_curled and pinky_curled and thumb_folded


def map_coordinates_to_screen(norm_x, norm_y, screen_w, screen_h, zone_left=0.20, zone_right=0.80, zone_top=0.15, zone_bottom=0.70):
    """
    Linearly maps normalized camera coordinates within an active interaction zone
    to screen pixel coordinates, with boundary clamping.
    """
    # Scale from [zone_left, zone_right] -> [0, screen_w]
    screen_x = np.interp(norm_x, [zone_left, zone_right], [0, screen_w])
    # Scale from [zone_top, zone_bottom] -> [0, screen_h]
    screen_y = np.interp(norm_y, [zone_top, zone_bottom], [0, screen_h])

    # Clamp safely inside screen bounds
    clamped_x = int(np.clip(screen_x, 0, screen_w - 1))
    clamped_y = int(np.clip(screen_y, 0, screen_h - 1))
    return clamped_x, clamped_y
