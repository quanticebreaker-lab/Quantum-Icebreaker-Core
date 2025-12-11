# pendulum_V58.py
"""
Pendulum Tracker V58 - Phase Space Stabilization
- FIX 1: Applied SmartLimiter to Phase Plot axes.
         (Prevents axes from "jittering" or shrinking when the spiral gets smaller).
- FIX 2: Increased Velocity Smoothing (EMA 0.02).
         (Makes the Phase Portrait line smooth and round, removing "broken" look).
- LAYOUT: Vertical Stack (V56 style).
"""
import cv2
import numpy as np
import time
import csv
import collections
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
from datetime import datetime
import threading
import math
import json
import os
import argparse

try:
    matplotlib.use('TkAgg')
except:
    pass

# ----------------------------
# Configuration
# ----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30.0
TARGET_FRAME_TIME = 1.0 / TARGET_FPS

# --- ROI Settings ---
ROI_PADDING_X = 180
ROI_PADDING_Y = 90
ROI_LOST_THRESHOLD = 10 
RECOVERY_WINDOW_SCALE = 0.5 

# --- OBS Dashboard File Names ---
FREQ_FILE = "obs_frequency.txt"
AMP_X_FILE = "obs_amplitude_x.txt"
FPS_FILE = "obs_fps.txt"

CSV_PREFIX = "pendulum"
MIN_CONTOUR_AREA = 80
MAXLEN_FOR_ANALYSIS = 1200

# --- VISUAL TUNING ---
PLOT_REFRESH_INTERVAL = 0.05   
EMA_ALPHA_POS = 0.08           # Position smoothing
EMA_ALPHA_VEL = 0.02           # <--- VELOCITY SMOOTHING (Lower = Smoother Phase Plot)

# --- SETTINGS FOR GRAPHS ---
TRAJECTORY_PLOT_LEN = 200      
PHASE_HISTORY_SECONDS = 2.0    # Show last 2 seconds on Phase Plot

PEAK_PROMINENCE = 4
PEAK_DISTANCE_FRAMES = 20
CALIB_FILENAME = "calibration.json"

TRAJECTORY_LEN = 30            
MAX_SPEED_COLOR_REF = 40.0     

# ----------------------------
# Helper: Online EMA Filter
# ----------------------------
class OnlineEMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last_val = None

    def update(self, measurement):
        if math.isnan(measurement):
            return self.last_val
        if self.last_val is None:
            self.last_val = measurement
        else:
            self.last_val = self.alpha * measurement + (1 - self.alpha) * self.last_val
        return self.last_val

# ----------------------------
# Helper Class: Smart Axis Stabilizer
# ----------------------------
class SmartLimiter:
    """
    Stabilizes graph axes. Expands instantly, shrinks very slowly.
    Prevents the "jittering/breathing" effect.
    """
    def __init__(self, decay=0.995, pad=0.1, min_range=10, is_centered=False):
        self.min_val = -min_range/2 if is_centered else 0
        self.max_val = min_range/2 if is_centered else min_range
        self.decay = decay
        self.pad = pad
        self.min_range = min_range

    def update(self, data_array):
        # Filter NaNs
        valid = np.array(data_array)
        valid = valid[~np.isnan(valid)]
        
        if len(valid) == 0: 
            return (self.min_val, self.max_val)
        
        c_min = np.min(valid)
        c_max = np.max(valid)
        
        # Expand instantly
        if c_max > self.max_val: self.max_val = c_max
        if c_min < self.min_val: self.min_val = c_min
        
        # Shrink slowly (decay towards current extremes)
        self.max_val = self.max_val * self.decay + c_max * (1 - self.decay)
        self.min_val = self.min_val * self.decay + c_min * (1 - self.decay)
        
        # Ensure min range
        if (self.max_val - self.min_val) < self.min_range:
            mid = (self.max_val + self.min_val) / 2
            self.max_val = mid + self.min_range / 2
            self.min_val = mid - self.min_range / 2
            
        span = self.max_val - self.min_val
        return (self.min_val - span*self.pad, self.max_val + span*self.pad)

# ----------------------------
# Data Sharing Class
# ----------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_to_show = None
        self.running = True
        self.calibration_done = False
        
        self.times = []
        self.smooth_x = []
        self.smooth_y = []
        self.smooth_vx = []
        
        self.lower1 = np.array([0, 0, 0])
        self.upper1 = np.array([0, 0, 0])
        self.lower2 = np.array([0, 0, 0])
        self.upper2 = np.array([0, 0, 0])
        
        self.show_rects = False 
        self.autoscale_y = True 

# ----------------------------
# Thread 1: Camera Capture
# ----------------------------
class CameraThread:
    def __init__(self, index=0, width=800, height=600, target_fps=25):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, int(target_fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            frame_copy = None if self.frame is None else self.frame.copy()
            grabbed = getattr(self, "grabbed", False)
        return grabbed, frame_copy

    def stop(self):
        self.running = False
        time.sleep(0.05)
        try: self.cap.release()
        except: pass

# ----------------------------
# Helpers
# ----------------------------
def init_kalman(start_x, start_y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.01 
    kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.05
    kf.statePost = np.array([start_x, start_y, 0, 0], np.float32)
    return kf

def get_color_by_speed(speed):
    ratio = min(speed / MAX_SPEED_COLOR_REF, 1.0)
    hue = (1.0 - ratio) * 120.0 
    color_hsv = np.uint8([[[hue/2, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

def nan_to_empty(x):
    return "" if (isinstance(x, float) and math.isnan(x)) else x

# ----------------------------
# Thread 2: Processing Worker
# ----------------------------
class ProcessingThread(threading.Thread):
    def __init__(self, cam, shared_state):
        super().__init__()
        self.cam = cam
        self.shared = shared_state
        self.daemon = True
        self.ema_x = OnlineEMA(alpha=EMA_ALPHA_POS)
        self.ema_y = OnlineEMA(alpha=EMA_ALPHA_POS)
        self.ema_vx = OnlineEMA(alpha=EMA_ALPHA_VEL)
        self.last_x = None
        self.last_t = None
        
    def run(self):
        while not self.shared.calibration_done and self.shared.running:
            time.sleep(0.1)
            
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"{CSV_PREFIX}_{now}.csv"
        csv_file = open(csv_filename, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["Time_s", "FrameCount", "X_px", "Y_px", "Amp_X", "Freq_Hz", "FPS"])
        
        smooth_hist_x = collections.deque(maxlen=MAXLEN_FOR_ANALYSIS)
        smooth_hist_y = collections.deque(maxlen=MAXLEN_FOR_ANALYSIS)
        smooth_hist_vx = collections.deque(maxlen=MAXLEN_FOR_ANALYSIS)
        times = collections.deque(maxlen=MAXLEN_FOR_ANALYSIS)
        trajectory_points = collections.deque(maxlen=TRAJECTORY_LEN)
        kf = None
        current_roi_rect = None
        lost_tracker_count = 0
        frame_count = 0
        
        ret, test_frame = self.cam.read()
        while not ret and self.shared.running:
            time.sleep(0.01)
            ret, test_frame = self.cam.read()
            
        ACTUAL_HEIGHT, ACTUAL_WIDTH, _ = test_frame.shape
        rec_w = int(ACTUAL_WIDTH * RECOVERY_WINDOW_SCALE)
        rec_h = int(ACTUAL_HEIGHT * RECOVERY_WINDOW_SCALE)
        rec_w = rec_w if rec_w % 2 == 0 else rec_w - 1
        rec_h = rec_h if rec_h % 2 == 0 else rec_h - 1
        rec_x = int((ACTUAL_WIDTH - rec_w) / 2)
        rec_y = int((ACTUAL_HEIGHT - rec_h) / 2)
        RECOVERY_RECT = (rec_x, rec_y, rec_w, rec_h)
        
        start_time = time.time()
        last_loop_time = time.time()
        fps_smooth = TARGET_FPS

        print(f"Worker Thread Started.")

        while self.shared.running:
            loop_start = time.time()
            ret, frame = self.cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            
            frame_count += 1
            t = time.time() - start_time
            
            # --- Vision & Kalman ---
            predicted_point = None
            if kf is not None:
                prediction = kf.predict()
                pred_x, pred_y = prediction[0], prediction[1]
                predicted_point = (int(pred_x), int(pred_y))

            roi_offset = (0, 0)
            if current_roi_rect:
                x, y, w, h = current_roi_rect
                if predicted_point is not None:
                    px, py = predicted_point
                    x = max(0, min(px - ROI_PADDING_X, ACTUAL_WIDTH - w))
                    y = max(0, min(py - ROI_PADDING_Y, ACTUAL_HEIGHT - h))
                frame_roi = frame[y:y+h, x:x+w]
                if frame_roi.size == 0:
                    current_roi_rect = None
                    continue
                frame_proc = cv2.convertScaleAbs(frame_roi, alpha=1.0, beta=0)
                roi_offset = (x, y)
                if self.shared.show_rects:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 1)
            else:
                x, y, w, h = RECOVERY_RECT
                frame_roi = frame[y:y+h, x:x+w]
                if frame_roi.size == 0: continue
                frame_proc = cv2.convertScaleAbs(frame_roi, alpha=1.0, beta=0)
                roi_offset = (x, y)
                if self.shared.show_rects:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            blurred = cv2.GaussianBlur(frame_proc, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            with self.shared.lock:
                l1, u1 = self.shared.lower1, self.shared.upper1
                l2, u2 = self.shared.lower2, self.shared.upper2
                
            mask = cv2.inRange(hsv, l1, u1)
            if np.any(l2):
                mask2 = cv2.inRange(hsv, l2, u2)
                mask = cv2.bitwise_or(mask, mask2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=3)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pendulum_x = float("nan")
            pendulum_y = float("nan")
            object_found = False

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        object_found = True
                        cx_roi = float(M["m10"] / M["m00"])
                        cy_roi = float(M["m01"] / M["m00"])
                        pendulum_x = cx_roi + roi_offset[0]
                        pendulum_y = cy_roi + roi_offset[1]
                        
                        if kf is None:
                            kf = init_kalman(pendulum_x, pendulum_y)
                        else:
                            measurement = np.array([[np.float32(pendulum_x)], [np.float32(pendulum_y)]])
                            kf.correct(measurement)
                        
                        w_roi = ROI_PADDING_X * 2
                        h_roi = ROI_PADDING_Y * 2
                        x_roi = max(0, int(pendulum_x) - ROI_PADDING_X)
                        y_roi = max(0, int(pendulum_y) - ROI_PADDING_Y)
                        if x_roi + w_roi > ACTUAL_WIDTH: x_roi = ACTUAL_WIDTH - w_roi
                        if y_roi + h_roi > ACTUAL_HEIGHT: y_roi = ACTUAL_HEIGHT - h_roi
                        current_roi_rect = (x_roi, y_roi, w_roi, h_roi)
                        lost_tracker_count = 0
                        
                        trajectory_points.append((int(pendulum_x), int(pendulum_y)))
                        cv2.circle(frame, (int(pendulum_x), int(pendulum_y)), 5, (0, 0, 255), -1)

            if not object_found:
                if current_roi_rect:
                    lost_tracker_count += 1
                    if predicted_point is not None and lost_tracker_count < ROI_LOST_THRESHOLD:
                         trajectory_points.append(predicted_point)
                    if lost_tracker_count > ROI_LOST_THRESHOLD:
                        current_roi_rect = None
                        lost_tracker_count = 0
                        kf = None
                        trajectory_points.clear()

            if len(trajectory_points) > 7:
                pts = list(trajectory_points)
                raw_speeds = []
                for i in range(1, len(pts)):
                    d = math.sqrt((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)
                    raw_speeds.append(d)
                smoothed_speeds = []
                w = 3 
                for i in range(len(raw_speeds)):
                    s = max(0, i-w)
                    e = min(len(raw_speeds), i+w+1)
                    chunk = raw_speeds[s:e]
                    smoothed_speeds.append(sum(chunk)/len(chunk) if chunk else 0)
                for i in range(1, len(pts)):
                    idx = i-1
                    if idx < len(smoothed_speeds):
                        color = get_color_by_speed(smoothed_speeds[idx])
                        thickness = int(np.sqrt(15 * float(i) / len(pts))) or 1
                        cv2.line(frame, pts[i-1], pts[i], color, thickness, lineType=cv2.LINE_AA)

            # Online Smoothing
            val_x = self.ema_x.update(pendulum_x)
            val_y = self.ema_y.update(pendulum_y)
            
            vx_val = float("nan")
            if val_x is not None and self.last_x is not None:
                dt = t - self.last_t if self.last_t is not None else 0.033
                if dt > 0:
                    raw_vx = (val_x - self.last_x) / dt
                    vx_val = self.ema_vx.update(raw_vx)
                
            if val_x is not None:
                self.last_x = val_x
                self.last_t = t

            safe_x = val_x if val_x is not None else float("nan")
            safe_y = val_y if val_y is not None else float("nan")
            safe_vx = vx_val if vx_val is not None else float("nan")
            
            smooth_hist_x.append(safe_x)
            smooth_hist_y.append(safe_y)
            smooth_hist_vx.append(safe_vx)
            times.append(t)
            
            amplitude_x = float("nan")
            frequency = float("nan")
            
            if frame_count % 5 == 0 and len(smooth_hist_x) >= 40:
                try:
                    recent_len = min(len(smooth_hist_x), 300)
                    pos_arr_x = np.array(list(smooth_hist_x)[-recent_len:])
                    valid_mask = ~np.isnan(pos_arr_x)
                    if np.sum(valid_mask) > 0:
                        valid_data = pos_arr_x[valid_mask]
                        peaks, _ = find_peaks(valid_data, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE_FRAMES)
                        recent_times = np.array(list(times)[-recent_len:])
                        valid_times = recent_times[valid_mask]
                        if len(peaks) >= 2:
                            t_peaks = valid_times[peaks]
                            diffs = np.diff(t_peaks)
                            if diffs.size > 0: frequency = 1.0 / np.mean(diffs[-5:])
                        amplitude_x = float(np.max(valid_data) - np.min(valid_data))
                except: pass

            now_loop = time.time()
            loop_dt = now_loop - last_loop_time if last_loop_time is not None else 1.0 / TARGET_FPS
            last_loop_time = now_loop
            instant_fps = 1.0 / loop_dt if loop_dt > 1e-6 else TARGET_FPS
            fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1

            if frame_count % 10 == 0: 
                try:
                    with open(FREQ_FILE, "w") as f: f.write(f"Freq: {frequency:.3f} Hz" if not math.isnan(frequency) else "Freq: --- Hz")
                    with open(AMP_X_FILE, "w") as f: f.write(f"Amp X: {amplitude_x:.0f} px" if not math.isnan(amplitude_x) else "Amp X: --- px")
                    with open(FPS_FILE, "w") as f: f.write(f"{fps_smooth:.1f} FPS")
                except: pass

            writer.writerow([f"{t:.4f}", frame_count, nan_to_empty(pendulum_x), nan_to_empty(pendulum_y), nan_to_empty(amplitude_x), nan_to_empty(frequency), f"{fps_smooth:.2f}"])

            right_align_x = FRAME_WIDTH - 280
            cv2.putText(frame, f"FPS: {int(fps_smooth)}", (right_align_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 0), 2)
            status_text = "Scale: AUTO" if self.shared.autoscale_y else "Scale: LOCKED"
            status_color = (0, 255, 0) if self.shared.autoscale_y else (0, 0, 255)
            cv2.putText(frame, status_text, (right_align_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, "'d': Toggle Box | 'a': Lock Scale | 'q': Quit", (20, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            with self.shared.lock:
                self.shared.frame_to_show = frame.copy()
                self.shared.times = list(times)
                self.shared.smooth_x = list(smooth_hist_x)
                self.shared.smooth_y = list(smooth_hist_y)
                self.shared.smooth_vx = list(smooth_hist_vx)
            
            elapsed = time.time() - loop_start
            time.sleep(max(0, TARGET_FRAME_TIME - elapsed))

        csv_file.close()

# ----------------------------
# UI / Calibration
# ----------------------------
def make_trackbar_window(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 900, 700)
    cv2.createTrackbar("LowH1", name, 0, 179, lambda x: None)
    cv2.createTrackbar("HighH1", name, 10, 179, lambda x: None)
    cv2.createTrackbar("LowH2", name, 160, 179, lambda x: None)
    cv2.createTrackbar("HighH2", name, 179, 179, lambda x: None)
    cv2.createTrackbar("LowS", name, 100, 255, lambda x: None)
    cv2.createTrackbar("HighS", name, 255, 255, lambda x: None)
    cv2.createTrackbar("LowV", name, 100, 255, lambda x: None)
    cv2.createTrackbar("HighV", name, 255, 255, lambda x: None)

def read_trackbar_values(name):
    return (cv2.getTrackbarPos("LowH1", name), cv2.getTrackbarPos("HighH1", name),
            cv2.getTrackbarPos("LowH2", name), cv2.getTrackbarPos("HighH2", name),
            cv2.getTrackbarPos("LowS", name), cv2.getTrackbarPos("HighS", name),
            cv2.getTrackbarPos("LowV", name), cv2.getTrackbarPos("HighV", name))

def calibration_loop(cam, shared):
    calib_win = "Color Calibration"
    make_trackbar_window(calib_win)
    loaded = None
    if os.path.exists(CALIB_FILENAME):
        try:
            with open(CALIB_FILENAME, "r") as f:
                d = json.load(f)
                loaded = (np.array(d["lower1"]), np.array(d["upper1"]), np.array(d["lower2"]), np.array(d["upper2"]))
        except: pass
        
    if loaded:
        l1, u1, l2, u2 = loaded
        cv2.setTrackbarPos("LowH1", calib_win, int(l1[0]))
        cv2.setTrackbarPos("HighH1", calib_win, int(u1[0]))
        cv2.setTrackbarPos("LowH2", calib_win, int(l2[0]))
        cv2.setTrackbarPos("HighH2", calib_win, int(u2[0]))
        cv2.setTrackbarPos("LowS", calib_win, int(l1[1]))
        cv2.setTrackbarPos("HighS", calib_win, int(u1[1]))
        cv2.setTrackbarPos("LowV", calib_win, int(l1[2]))
        cv2.setTrackbarPos("HighV", calib_win, int(u1[2]))

    print("Adjust sliders. Press 's' to save. 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        calib_preview = cv2.resize(frame, (800, 450))
        blurred = cv2.GaussianBlur(calib_preview, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lh1, hh1, lh2, hh2, ls, hs, lv, hv = read_trackbar_values(calib_win)
        l1 = np.array([lh1, ls, lv], dtype=np.int32)
        u1 = np.array([hh1, hs, hv], dtype=np.int32)
        l2 = np.array([lh2, ls, lv], dtype=np.int32)
        u2 = np.array([hh2, hs, hv], dtype=np.int32)
        mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
        combined = np.hstack([calib_preview, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(calib_win, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            shared.lower1, shared.upper1 = l1, u1
            shared.lower2, shared.upper2 = l2, u2
            with open(CALIB_FILENAME, "w") as f:
                json.dump({"lower1":l1.tolist(),"upper1":u1.tolist(),"lower2":l2.tolist(),"upper2":u2.tolist()}, f)
            cv2.destroyWindow(calib_win)
            shared.calibration_done = True
            break
        elif key == ord('q'):
            shared.running = False
            cv2.destroyAllWindows()
            return

# ----------------------------
# Main
# ----------------------------
def run():
    shared = SharedState()
    cam = CameraThread(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS).start()
    calibration_loop(cam, shared)
    if not shared.running:
        cam.stop()
        return

    worker = ProcessingThread(cam, shared)
    worker.start()
    
    meas_win = "Pendulum V58 (Stable Phase)"
    cv2.namedWindow(meas_win, cv2.WINDOW_NORMAL)
    cv2.imshow(meas_win, np.zeros((540, 960, 3), dtype=np.uint8))
    cv2.resizeWindow(meas_win, 960, 540)
    cv2.moveWindow(meas_win, 0, 0)
    cv2.waitKey(1)
    
    plt.ion()
    fig = plt.figure(figsize=(6, 12), constrained_layout=True)
    
    def on_key(event):
        if event.key == 'q': shared.running = False
        elif event.key == 'd': shared.show_rects = not shared.show_rects
        elif event.key == 'a': shared.autoscale_y = not shared.autoscale_y
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, "window"):
             try: mngr.window.wm_geometry("+960+0")
             except: pass
    except: pass

    # Layout: 3 Rows
    gs = fig.add_gridspec(3, 1)
    
    ax_time = fig.add_subplot(gs[0, 0])
    line_time, = ax_time.plot([], [], 'b-', linewidth=1.5)
    ax_time.set_ylabel("X Position")
    ax_time.set_title("Time Domain")
    ax_time.grid(True)
    
    ax_traj = fig.add_subplot(gs[1, 0])
    line_traj, = ax_traj.plot([], [], 'g-', linewidth=1.5)
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_title("Trajectory")
    ax_traj.grid(True)
    ax_traj.invert_yaxis()

    ax_phase = fig.add_subplot(gs[2, 0])
    line_phase, = ax_phase.plot([], [], 'r-', linewidth=1.5)
    ax_phase.set_xlabel("X")
    ax_phase.set_ylabel("Velocity")
    ax_phase.set_title("Phase Space")
    ax_phase.grid(True)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)

    # Initialize Smart Limiters
    phase_x_limiter = SmartLimiter(decay=0.9, pad=0.15, min_range=100)
    phase_v_limiter = SmartLimiter(decay=0.95, pad=0.15, min_range=100, is_centered=True) # Velocity centered at 0

    last_plot_time = time.time()
    
    print("\n" + "="*50)
    print("V58 Active - Phase Stabilization")
    print(f" - Phase History: {PHASE_HISTORY_SECONDS}s")
    print(" - Smart Scaling: ON (No jitter)")
    print("="*50 + "\n")

    try:
        while shared.running:
            frame = None
            with shared.lock:
                if shared.frame_to_show is not None:
                    frame = shared.frame_to_show.copy()
            if frame is not None:
                cv2.imshow(meas_win, cv2.resize(frame, (960, 540)))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): shared.running = False; break
            elif key == ord('d'): shared.show_rects = not shared.show_rects
            elif key == ord('a'): shared.autoscale_y = not shared.autoscale_y
            
            if time.time() - last_plot_time > PLOT_REFRESH_INTERVAL:
                last_plot_time = time.time()
                
                with shared.lock:
                    raw_ts = list(shared.times)
                    xs_data = list(shared.smooth_x)
                    ys_data = list(shared.smooth_y)
                    vx_data = list(shared.smooth_vx)
                
                if len(raw_ts) > 2:
                    ts_arr = np.array(raw_ts)
                    try:
                        # 1. Update Data (ALWAYS)
                        t_end = ts_arr[-1]
                        t_start = max(ts_arr[0], t_end - 10.0) 
                        line_time.set_data(ts_arr, xs_data)
                        ax_time.set_xlim(t_start, t_end + 0.2) 
                        
                        tail_traj = min(len(xs_data), TRAJECTORY_PLOT_LEN)
                        d_x_traj = xs_data[-tail_traj:]
                        d_y_traj = ys_data[-tail_traj:]
                        line_traj.set_data(d_x_traj, d_y_traj)
                        
                        # Slice Phase Data
                        t_cutoff = t_end - PHASE_HISTORY_SECONDS
                        # Approximate slice
                        tail_len_p = int(PHASE_HISTORY_SECONDS * TARGET_FPS * 1.5)
                        tail_len_p = min(len(ts_arr), tail_len_p)
                        t_tail = ts_arr[-tail_len_p:]
                        mask_p = t_tail > t_cutoff
                        
                        d_x_phase = np.array(xs_data)[-tail_len_p:][mask_p]
                        d_vx_phase = np.array(vx_data)[-tail_len_p:][mask_p]
                        line_phase.set_data(d_x_phase, d_vx_phase)
                        
                        # 2. Update Scales (Only if Autoscale ON)
                        if shared.autoscale_y:
                            # Time
                            visible_mask = (ts_arr >= t_start)
                            if np.any(visible_mask):
                                vis_y = np.array(xs_data)[visible_mask]
                                vis_y = vis_y[~np.isnan(vis_y)]
                                if len(vis_y) > 0:
                                    mn, mx = np.min(vis_y), np.max(vis_y)
                                    pad = max(10, (mx-mn)*0.1)
                                    ax_time.set_ylim(mn-pad, mx+pad)
                            
                            # Trajectory
                            d_x_v = np.array(d_x_traj)
                            d_y_v = np.array(d_y_traj)
                            valid_t = ~np.isnan(d_x_v) & ~np.isnan(d_y_v)
                            if np.any(valid_t):
                                mn_x, mx_x = np.min(d_x_v[valid_t]), np.max(d_x_v[valid_t])
                                mn_y, mx_y = np.min(d_y_v[valid_t]), np.max(d_y_v[valid_t])
                                pad_x = max(10, (mx_x-mn_x)*0.1)
                                pad_y = max(10, (mx_y-mn_y)*0.1)
                                ax_traj.set_xlim(mn_x-pad_x, mx_x+pad_x)
                                ax_traj.set_ylim(mx_y+pad_y, mn_y-pad_y)
                                
                            # Phase Space (Use SmartLimiter for stability)
                            px_lim = phase_x_limiter.update(d_x_phase)
                            pv_lim = phase_v_limiter.update(d_vx_phase)
                            ax_phase.set_xlim(px_lim)
                            ax_phase.set_ylim(pv_lim)

                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    except: pass
            
            time.sleep(0.001)

    finally:
        shared.running = False
        cam.stop()
        cv2.destroyAllWindows()
        plt.close('all')
        
        try:
            with open(FREQ_FILE, "w") as f: f.write("OFFLINE")
            with open(AMP_X_FILE, "w") as f: f.write("OFFLINE")
            with open(FPS_FILE, "w") as f: f.write("OFFLINE")
        except: pass
        print("Exited.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()
    CAMERA_INDEX = args.camera
    run()