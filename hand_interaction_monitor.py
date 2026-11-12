"""
CCTV Hand Interaction Monitor — Jewellery Store
================================================
Detects when a staff member hands an object to a customer by tracking
hand/wrist proximity across two user-defined ROIs.

Usage:
    python hand_interaction_monitor.py

Controls (ROI Drawing Phase):
    Left-click  → add polygon point
    Right-click → finish current polygon early
    R           → reset and redraw both ROIs
    S           → save ROIs and start monitoring
    Q           → quit

Controls (Monitoring Phase):
    R           → re-draw ROIs
    +/-         → increase/decrease proximity threshold
    Q           → quit

Dependencies:
    pip install ultralytics opencv-python numpy winsound
"""

import cv2
import json
import math
import os
import time
import winsound
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# NOTE: The '@' in the password must be URL-encoded as '%40' so OpenCV/FFMPEG
#       can correctly parse the RTSP URL.
# ─────────────────────────────────────────────────────────────────────────────
RTSP_URL = "rtsp://admin:Jezt%40884@192.168.1.108/cam/realmonitor?channel=1&subtype=0"
MODEL_PATH = "yolov8m-pose.pt"          # Downloaded automatically by Ultralytics
CONFIG_FILE = "roi_config.json"
ALERTS_DIR = Path("alerts")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Alert cooldown — seconds between successive alerts
ALERT_COOLDOWN = 5.0

# Default proximity threshold in pixels (at 1280×720 resolution)
DEFAULT_THRESHOLD = 180

# YOLOv8-Pose keypoint indices (COCO 17-point skeleton)
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10
KP_LEFT_ELBOW  = 7   # left elbow (fallback if wrist not visible)
KP_RIGHT_ELBOW = 8   # right elbow (fallback)

# Confidence threshold for keypoints to be considered valid
KP_CONF_THRESHOLD = 0.35

# ROI zone colors  (BGR)
COLOR_STAFF    = (0,  80, 220)
COLOR_CUSTOMER = (220, 130,  0)
COLOR_ALERT    = (0,   0, 255)
COLOR_OK       = (0, 220,  60)
COLOR_LINE     = (0, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def point_in_polygon(point, polygon):
    """Ray-casting algorithm for point-in-polygon test."""
    if len(polygon) < 3:
        return False
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[-1]
    for cx, cy in polygon:
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-9) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


def dist2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_filled_polygon(frame, pts, color, alpha=0.25):
    """Draw a semi-transparent filled polygon overlay."""
    if len(pts) < 3:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, 2)


def draw_label(frame, text, pos, font_scale=0.6, thickness=1,
               text_color=(255, 255, 255), bg_color=(30, 30, 30)):
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


def save_config(staff_roi, customer_roi, threshold):
    data = {
        "staff_roi": staff_roi,
        "customer_roi": customer_roi,
        "threshold": threshold,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Config saved to {CONFIG_FILE}")


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return None, None, DEFAULT_THRESHOLD
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    return data["staff_roi"], data["customer_roi"], data.get("threshold", DEFAULT_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
# ROI Drawing
# ─────────────────────────────────────────────────────────────────────────────

class ROIDrawer:
    """Interactive ROI polygon drawer. Collects 2 polygons from mouse clicks."""

    WIN_NAME = "ROI Setup"

    PHASE_LABELS = [
        "[1/2] DRAW STAFF ZONE - Left-click to place points",
        "[2/2] DRAW CUSTOMER ZONE - Left-click to place points",
    ]
    PHASE_COLORS = [COLOR_STAFF, COLOR_CUSTOMER]

    def __init__(self, frame):
        self.frame = frame.copy()
        self.polygons = [[], []]   # [staff_pts, customer_pts]
        self.phase = 0
        self.done = False

    def _mouse_cb(self, event, x, y, flags, param):
        if self.done or self.phase >= 2:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygons[self.phase].append([x, y])
            print(f"[ROI] Point added: ({x}, {y})  |  total={len(self.polygons[self.phase])}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._finish_phase()

    def _finish_phase(self):
        if len(self.polygons[self.phase]) >= 3:
            label = ["STAFF", "CUSTOMER"][self.phase]
            print(f"[ROI] {label} zone saved with {len(self.polygons[self.phase])} points.")
            self.phase += 1
            if self.phase >= 2:
                self.done = True
        else:
            print(f"[ROI] Need at least 3 points (have {len(self.polygons[self.phase])}).")

    def run(self):
        win = self.WIN_NAME  # short alias
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(win, self._mouse_cb)

        print("[INFO] ROI window opened.")
        print("[INFO]   Left-click  = add point")
        print("[INFO]   S key       = finish current zone (need >= 3 points)")
        print("[INFO]   Right-click = finish current zone")
        print("[INFO]   R key       = reset everything")
        print("[INFO]   Q key       = quit")

        while not self.done:
            display = self.frame.copy()

            # Draw completed polygons
            for i in range(self.phase):
                draw_filled_polygon(display, self.polygons[i], self.PHASE_COLORS[i])
                if self.polygons[i]:
                    label = ["STAFF", "CUSTOMER"][i]
                    lx = int(np.mean([p[0] for p in self.polygons[i]]))
                    ly = int(np.mean([p[1] for p in self.polygons[i]]))
                    draw_label(display, label, (lx - 30, ly),
                               font_scale=0.8, bg_color=self.PHASE_COLORS[i])

            # Draw current in-progress polygon
            if self.phase < 2:
                cur = self.polygons[self.phase]
                color = self.PHASE_COLORS[self.phase]
                for pt in cur:
                    cv2.circle(display, tuple(pt), 6, color, -1)
                    cv2.circle(display, tuple(pt), 6, (255, 255, 255), 1)
                if len(cur) > 1:
                    cv2.polylines(display, [np.array(cur, dtype=np.int32)], False, color, 2)

            # Instruction overlay
            if self.phase < 2:
                cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 70), (20, 20, 20), -1)
                cv2.putText(display, self.PHASE_LABELS[self.phase],
                            (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                            self.PHASE_COLORS[self.phase], 2, cv2.LINE_AA)
                cv2.putText(display,
                            f"Points: {len(self.polygons[self.phase])}  (need >=3)   "
                            "S or right-click=finish   R=reset   Q=quit",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win, display)

            # Bring window to front once
            if not hasattr(self, '_front'):
                self._front = True
                try:
                    import ctypes
                    hwnd = ctypes.windll.user32.FindWindowW(None, win)
                    if hwnd:
                        ctypes.windll.user32.ShowWindow(hwnd, 9)
                        ctypes.windll.user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass

            key = cv2.waitKey(30) & 0xFF

            if key == 255:   # no key pressed
                continue
            elif key in (ord('s'), ord('S')):
                self._finish_phase()
            elif key in (ord('r'), ord('R')):
                self.polygons = [[], []]
                self.phase = 0
                print("[ROI] Reset. Draw STAFF zone first.")
            elif key in (ord('q'), ord('Q'), 27):   # Q or Esc
                cv2.destroyWindow(win)
                return None, None

        cv2.destroyWindow(win)
        return self.polygons[0], self.polygons[1]


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton drawing
# ─────────────────────────────────────────────────────────────────────────────

# COCO upper-body skeleton connections
_SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),   # L shoulder → elbow → wrist
    (6, 8), (8, 10),  # R shoulder → elbow → wrist
    (5, 6),           # shoulders
    (5, 11), (6, 12), # shoulders → hips
    (11, 12),         # hips
    (0, 5), (0, 6),   # nose → shoulders
]


def draw_skeleton(frame, kps, color):
    for a, b in _SKELETON_CONNECTIONS:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca >= KP_CONF_THRESHOLD and cb >= KP_CONF_THRESHOLD:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for kx, ky, kconf in kps:
        if kconf >= KP_CONF_THRESHOLD:
            cv2.circle(frame, (int(kx), int(ky)), 4, (255, 255, 255), -1)
            cv2.circle(frame, (int(kx), int(ky)), 4, color, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────────────────

def build_annotated_frame(frame, staff_roi, customer_roi,
                           staff_boxes, customer_boxes,
                           staff_hands, customer_hands,
                           closest_pair, min_dist, threshold,
                           alert, fps):
    out = frame.copy()

    # ROI overlays
    draw_filled_polygon(out, staff_roi,    COLOR_STAFF,    alpha=0.20)
    draw_filled_polygon(out, customer_roi, COLOR_CUSTOMER, alpha=0.20)

    # ROI labels
    if staff_roi:
        sx = int(np.mean([p[0] for p in staff_roi]))
        sy = int(np.mean([p[1] for p in staff_roi]))
        draw_label(out, "STAFF ZONE",    (sx - 50, sy), font_scale=0.7, bg_color=COLOR_STAFF)
    if customer_roi:
        cx2 = int(np.mean([p[0] for p in customer_roi]))
        cy2 = int(np.mean([p[1] for p in customer_roi]))
        draw_label(out, "CUSTOMER ZONE", (cx2 - 60, cy2), font_scale=0.7, bg_color=COLOR_CUSTOMER)

    # Bounding boxes + skeletons
    for (box, kps) in staff_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_STAFF, 2)
        draw_label(out, "STAFF", (x1, y1 - 8), font_scale=0.55, bg_color=COLOR_STAFF)
        draw_skeleton(out, kps, COLOR_STAFF)
    for (box, kps) in customer_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_CUSTOMER, 2)
        draw_label(out, "CUSTOMER", (x1, y1 - 8), font_scale=0.55, bg_color=COLOR_CUSTOMER)
        draw_skeleton(out, kps, COLOR_CUSTOMER)

    # Hand marker dots
    for sh in staff_hands:
        cv2.circle(out, sh, 9, COLOR_STAFF,    -1)
        cv2.circle(out, sh, 9, (255,255,255),   1)
    for ch in customer_hands:
        cv2.circle(out, ch, 9, COLOR_CUSTOMER, -1)
        cv2.circle(out, ch, 9, (255,255,255),   1)

    # Distance lines between all staff and customer hands (bipartite graph)
    if staff_hands and customer_hands:
        for sh in staff_hands:
            for ch in customer_hands:
                d = dist2d(sh, ch)
                # If this specific pair is too close, draw it red and thicker
                line_color = COLOR_ALERT if d < threshold else COLOR_LINE
                thickness = 2 if d < threshold else 1
                cv2.line(out, sh, ch, line_color, thickness, cv2.LINE_AA)
        
        # To avoid text clutter, only draw the numeric distance label on the absolute closest pair
        if closest_pair:
            sh, ch = closest_pair
            mid = (int((sh[0] + ch[0]) / 2), int((sh[1] + ch[1]) / 2))
            draw_label(out, f"Min: {int(min_dist)} px", (mid[0], mid[1] - 12),
                       font_scale=0.55, bg_color=(40, 40, 40))

    # Bottom status bar
    h, w = out.shape[:2]
    bar_h = 38
    cv2.rectangle(out, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    status = (
        f"Threshold: {threshold}px  |  "
        f"Staff hands: {len(staff_hands)}  |  Customer hands: {len(customer_hands)}"
    )
    if staff_hands and customer_hands:
        status += f"  |  Min dist: {int(min_dist)}px"
    status += f"  |  FPS: {fps}  |  [+/-] threshold  [R] re-draw  [Q] quit"
    cv2.putText(out, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    # Alert banner
    if alert:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 72), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
        cv2.putText(out,
                    "  !! ALERT - HAND CONTACT / OBJECT TRANSFER DETECTED !!",
                    (14, 46), cv2.FONT_HERSHEY_DUPLEX,
                    0.95, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main monitoring loop
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor(staff_roi, customer_roi, threshold):
    """
    Returns True  → caller should re-draw ROIs then call again.
    Returns False → normal exit.
    """
    ALERTS_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream. Check URL and network.")
        return False

    last_alert_time = 0.0
    alert_active    = False
    alert_end_time  = 0.0
    fps_counter     = 0
    fps_display     = 0
    fps_tick        = time.time()

    window = "CCTV Hand Interaction Monitor"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

    print("[INFO] Monitoring started.")
    print("[INFO] Press Q to quit | R to re-draw ROIs | +/- to adjust threshold")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed - reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        now   = time.time()

        # ── YOLOv8-Pose inference ─────────────────────────────────────
        results = model(frame, verbose=False, conf=0.35)[0]

        staff_hands    = []
        customer_hands = []
        staff_boxes    = []
        customer_boxes = []

        if results.keypoints is not None and results.boxes is not None:
            kps_data = results.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes    = results.boxes.xyxy.cpu().numpy()       # (N, 4)

            for kps, box in zip(kps_data, boxes):
                x1, y1, x2, y2 = box.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                in_staff    = point_in_polygon((cx, cy), staff_roi)
                in_customer = point_in_polygon((cx, cy), customer_roi)

                # Collect visible wrist keypoints (elbows excluded by user request)
                person_hand_pts = []
                for kp_idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
                    kx, ky, kconf = kps[kp_idx]
                    if kconf >= KP_CONF_THRESHOLD:
                        person_hand_pts.append((int(kx), int(ky)))

                if in_staff:
                    staff_hands.extend(person_hand_pts)
                    staff_boxes.append((box.astype(int), kps))
                elif in_customer:
                    customer_hands.extend(person_hand_pts)
                    customer_boxes.append((box.astype(int), kps))

        # ── Proximity check ───────────────────────────────────────────
        min_dist     = float("inf")
        closest_pair = None

        for sh in staff_hands:
            for ch in customer_hands:
                d = dist2d(sh, ch)
                if d < min_dist:
                    min_dist = d
                    closest_pair = (sh, ch)

        alert_triggered = (
            min_dist < threshold
            and len(staff_hands) > 0
            and len(customer_hands) > 0
            and (now - last_alert_time) > ALERT_COOLDOWN
        )

        if alert_triggered:
            last_alert_time = now
            alert_active    = True
            alert_end_time  = now + 3.0

            # Save annotated screenshot
            snapshot = build_annotated_frame(
                frame, staff_roi, customer_roi,
                staff_boxes, customer_boxes,
                staff_hands, customer_hands,
                closest_pair, min_dist, threshold,
                alert=True, fps=fps_display,
            )
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = ALERTS_DIR / f"alert_{ts}.jpg"
            cv2.imwrite(str(path), snapshot)
            print(f"[ALERT] Hand contact detected!  Screenshot → {path}")

            # Windows beep
            try:
                winsound.Beep(1200, 350)
                winsound.Beep(1000, 250)
            except Exception:
                pass

        if now > alert_end_time:
            alert_active = False

        # ── Draw & display ────────────────────────────────────────────
        display = build_annotated_frame(
            frame, staff_roi, customer_roi,
            staff_boxes, customer_boxes,
            staff_hands, customer_hands,
            closest_pair, min_dist, threshold,
            alert=alert_active, fps=fps_display,
        )

        fps_counter += 1
        if now - fps_tick >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_tick    = now

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            cap.release()
            cv2.destroyWindow(window)
            return True   # signal re-draw
        elif key in (ord('+'), ord('=')):
            threshold = min(threshold + 10, 400)
            save_config(staff_roi, customer_roi, threshold)
            print(f"[INFO] Threshold → {threshold} px")
        elif key in (ord('-'), ord('_')):
            threshold = max(threshold - 10, 20)
            save_config(staff_roi, customer_roi, threshold)
            print(f"[INFO] Threshold → {threshold} px")

    cap.release()
    cv2.destroyAllWindows()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def grab_frame_from_rtsp():
    """Grab a single representative frame from the RTSP stream."""
    print(f"[INFO] Connecting to RTSP stream…")
    # Use FFMPEG backend; set a short open timeout via env or options
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                          "rtsp_transport;tcp|analyzeduration;2000000|max_delay;500000")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERROR] cap.isOpened() returned False — check RTSP URL and network.")
        return None

    frame = None
    MAX_TRIES = 30   # try for ~3 seconds
    for attempt in range(MAX_TRIES):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            print(f"[INFO] Frame grabbed successfully (attempt {attempt + 1}).")
            break
        time.sleep(0.1)

    cap.release()

    if frame is None:
        print("[ERROR] Could not read a frame from RTSP after many attempts.")
        print("[ERROR] Possible causes:")
        print("        1. RTSP URL is wrong (check IP, port, credentials)")
        print("        2. Camera is offline")
        print("        3. Special characters in password need URL-encoding")
        print(f"        RTSP_URL used: {RTSP_URL}")

    return frame


def main():
    ALERTS_DIR.mkdir(exist_ok=True)

    staff_roi, customer_roi, threshold = load_config()

    # First run or user requested re-draw
    if not staff_roi or not customer_roi:
        print("[INFO] No saved ROI config — connecting to RTSP to grab reference frame…")
        frame = grab_frame_from_rtsp()
        if frame is None:
            print("[ERROR] Could not connect to RTSP stream. Check URL and network.")
            return
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        drawer = ROIDrawer(frame)
        staff_roi, customer_roi = drawer.run()
        if staff_roi is None:
            print("[INFO] Cancelled. Exiting.")
            return
        save_config(staff_roi, customer_roi, threshold)

    print(f"[INFO] Staff ROI    : {len(staff_roi)} vertices")
    print(f"[INFO] Customer ROI : {len(customer_roi)} vertices")
    print(f"[INFO] Threshold    : {threshold} px")

    while True:
        redraw = run_monitor(staff_roi, customer_roi, threshold)
        if not redraw:
            break

        # Re-draw ROIs
        print("[INFO] Re-drawing ROIs — grabbing reference frame…")
        frame = grab_frame_from_rtsp()
        if frame is None:
            print("[ERROR] Cannot grab frame. Exiting.")
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        drawer = ROIDrawer(frame)
        staff_roi, customer_roi = drawer.run()
        if staff_roi is None:
            break
        _, _, threshold = load_config()
        save_config(staff_roi, customer_roi, threshold)

    print("[INFO] Monitor stopped.")


if __name__ == "__main__":
    main()
