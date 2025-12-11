# compare_v21_first_peak_sync.py
"""
Pendulum Comparison Tool V21 (FIRST PEAK SYNC)
- BASED ON: V20.4
- FIX: Replaced FFT-based synchronization with First Peak Alignment.
       This ensures signals start at the same phase (first maximum) regardless of frequency differences.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider
from scipy.signal import medfilt, savgol_filter, find_peaks, fftconvolve
import os
import sys
import argparse
import logging

# ==========================================
# DEFAULT SETTINGS
# ==========================================
FILE_1_DEFAULT = "pendulum_2025-11-29_10-12-42.csv" 
FILE_2_DEFAULT = "pendulum_2025-11-29_10-19-33.csv"
FILE_3_DEFAULT = "pendulum_2025-11-29_10-25-32.csv"

INITIAL_WINDOW_SIZE = 20.0  
AUTO_SYNC_PHASE = True
RESAMPLE_DT = 0.033 
STOP_THRESHOLD_PX = 3.0 

# --- FILTERING ---
SG_POLYORDER = 3     
USE_MEDIAN_PREFILTER = True
MEDIAN_KERNEL = 5

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# ROBUST MATH UTILS
# ==========================================
def make_odd(n):
    n = int(n)
    if n < 1: n = 1
    if n % 2 == 0: n += 1
    return n

def safe_savgol(y, window, polyorder, dt=None):
    window = make_odd(window)
    if len(y) < window:
        if len(y) <= polyorder: return y 
        window = make_odd(len(y))
        if window <= polyorder: window = make_odd(polyorder + 1)
        if window > len(y): window = len(y) if len(y)%2!=0 else len(y)-1
    
    if window <= polyorder: return y 
    
    try:
        if dt is None:
            return savgol_filter(y, window, polyorder, deriv=0)
        else:
            return savgol_filter(y, window, polyorder, deriv=0, delta=dt)
    except Exception as e:
        logger.warning(f"Savgol failed: {e}")
        return y

def detrend_linear(sig):
    if len(sig) < 2: return sig
    t = np.arange(len(sig))
    p = np.polyfit(t, sig, 1)
    return sig - (p[0]*t + p[1])

def calculate_adaptive_window(dt):
    target_duration = 0.7
    window = int(round(target_duration / dt))
    window = make_odd(window)
    if window < 5: window = 5
    if window <= SG_POLYORDER: window = make_odd(SG_POLYORDER + 1)
    return window

def safe_interp(t, x, t_target):
    if len(t) == 0: return np.zeros_like(t_target)
    order = np.argsort(t)
    t_sorted, x_sorted = t[order], x[order]
    mask = ~np.isnan(x_sorted)
    if np.sum(mask) < 2: return np.full_like(t_target, np.nan, dtype=float)
    return np.interp(t_target, t_sorted[mask], x_sorted[mask])

# ==========================================
# DATA LOADING & PROCESSING
# ==========================================
def smart_find_column(header, candidates):
    header_lower = [h.lower().strip() for h in header]
    for candidate in candidates:
        for i, col_name in enumerate(header_lower):
            if candidate in col_name: return i
    return None

def load_data(filename):
    times, xs, ys = [], [], []
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return np.array([]), np.array([]), np.array([])
        
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_t = smart_find_column(header, ['time', 't_sec', 'seconds'])
            idx_x = smart_find_column(header, ['x_px', 'xpos', 'x_'])
            idx_y = smart_find_column(header, ['y_px', 'ypos', 'y_'])
            
            if idx_t is None: idx_t = 0
            if idx_x is None: idx_x = 2
            if idx_y is None: idx_y = 3
            
            for row in reader:
                try:
                    if len(row) > max(idx_t, idx_x, idx_y):
                        t_v, x_v, y_v = float(row[idx_t]), float(row[idx_x]), float(row[idx_y])
                        if not (np.isnan(t_v) or np.isnan(x_v)):
                            times.append(t_v); xs.append(x_v); ys.append(y_v)
                except ValueError: continue
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return np.array([]), np.array([]), np.array([])
            
    if not times: return np.array([]), np.array([]), np.array([])
    t_arr, x_arr, y_arr = np.array(times), np.array(xs), np.array(ys)
    sort_idx = np.argsort(t_arr)
    return t_arr[sort_idx], x_arr[sort_idx], y_arr[sort_idx]

def get_sampling_rate(t):
    if len(t) < 2: return 0.033
    dt = np.median(np.diff(t))
    return dt if dt > 0 else 0.033

def pre_calculate_peaks(x):
    if len(x) == 0: return np.array([], dtype=int)
    x_c = x - np.mean(x)
    std_val = np.std(x_c)
    if std_val == 0: return np.array([], dtype=int)
    peaks, _ = find_peaks(x_c, prominence=max(1e-6, std_val * 0.2))
    return peaks

def sync_by_first_peak(t1, x1, t2, x2):
    """
    Synchronize two signals by aligning their first detected peaks.
    """
    p1 = pre_calculate_peaks(x1)
    p2 = pre_calculate_peaks(x2)
    
    if len(p1) > 0 and len(p2) > 0:
        t_peak1 = t1[p1[0]]
        t_peak2 = t2[p2[0]]
        shift = t_peak2 - t_peak1
        return shift
        
    return 0.0

def calculate_damping_robust(t, x, peaks_idx, max_time=None):
    if len(peaks_idx) < 2: return 0.0, 0
    x_c = x - np.mean(x)
    t_peaks, x_peaks = t[peaks_idx], np.abs(x_c[peaks_idx])
    
    if max_time is not None:
        mask = t_peaks <= max_time
        t_peaks, x_peaks = t_peaks[mask], x_peaks[mask]
        
    mask_valid = x_peaks > 1e-6
    t_fit, amp_fit = t_peaks[mask_valid], x_peaks[mask_valid]
    if len(t_fit) < 3: return 0.0, len(t_fit)
    try:
        slope, _ = np.polyfit(t_fit, np.log(amp_fit), 1)
        return -slope, len(t_fit)
    except: return 0.0, len(t_fit)

def calculate_robust_period(t, peaks_idx, view_start, view_end, min_cycles=5):
    if len(peaks_idx) < 2: return None, 0
    t_peaks = t[peaks_idx]
    mask = (t_peaks >= view_start) & (t_peaks <= view_end)
    sel = t_peaks[mask]
    if len(sel) < 2:
        idx_closest = (np.abs(t_peaks - (view_start+view_end)/2.0)).argmin()
        sel = t_peaks[max(0, idx_closest - min_cycles//2) : min(len(t_peaks), idx_closest + min_cycles)]
    return (np.mean(np.diff(sel)), len(sel)) if len(sel) >= 2 else (None, 0)

def get_local_activity(t, x, start, end):
    mask = (t >= start) & (t < end)
    if not np.any(mask): return False, 0.0
    p2p = np.max(x[mask]) - np.min(x[mask])
    return p2p > STOP_THRESHOLD_PX, p2p

# ==========================================
# UI CLASS
# ==========================================
class TimeNavigator:
    def __init__(self, t_list, x_list, y_list, filenames):
        print(f"Initializing V21 UI...")
        self.ts, self.raw_xs, self.raw_ys, self.names = t_list, x_list, y_list, filenames
        self.xs, self.ys, self.vs, self.peaks, self.dampings = [], [], [], [], []
        self.min_duration, self.max_time = float('inf'), 0.0
        
        for i in range(3):
            t, x, y = self.ts[i], self.raw_xs[i], self.raw_ys[i]
            if len(t) == 0:
                self.xs.append(np.array([])); self.ys.append(np.array([])); self.vs.append(np.array([]))
                self.peaks.append(np.array([])); self.dampings.append((0.0, 0))
                continue
                
            self.max_time = max(self.max_time, t[-1])
            self.min_duration = min(self.min_duration, t[-1])
            dt = get_sampling_rate(t)
            win = calculate_adaptive_window(dt)
            
            if USE_MEDIAN_PREFILTER:
                k = make_odd(MEDIAN_KERNEL)
                if len(x) > k: x, y = medfilt(x, k), medfilt(y, k)
                    
            xf, yf = safe_savgol(x, win, SG_POLYORDER), safe_savgol(y, win, SG_POLYORDER)
            # Velocity calc
            if len(xf) > win:
                vf = savgol_filter(xf, win, SG_POLYORDER, deriv=1, delta=dt)
            else:
                vf = np.zeros_like(xf)

            self.xs.append(xf); self.ys.append(yf); self.vs.append(vf)
            pks = pre_calculate_peaks(xf)
            self.peaks.append(pks)
            self.dampings.append(calculate_damping_robust(t, xf, pks, max_time=None))

        if self.min_duration == float('inf'): self.min_duration = 0.0

        # Diff Matrix
        self.t_common = np.arange(0, self.max_time, RESAMPLE_DT)
        xi = [safe_interp(self.ts[i], self.xs[i], self.t_common) for i in range(3)]
        self.diffs = {
            '12': np.abs(xi[0]-xi[1]), '13': np.abs(xi[0]-xi[2]), '23': np.abs(xi[1]-xi[2])
        }
        for k in self.diffs:
            if len(self.diffs[k]) > 31: self.diffs[k] = savgol_filter(self.diffs[k], 31, 3)

        # PLOT STATE
        self.current_start = 0.0
        self.window_size = INITIAL_WINDOW_SIZE
        self.visible = [True, True, True]
        
        # --- LIMITS CALCULATION ---
        all_x = np.concatenate([x for x in self.xs if len(x)>0]) if any(len(x)>0 for x in self.xs) else np.array([0])
        all_y = np.concatenate([y for y in self.ys if len(y)>0]) if any(len(y)>0 for y in self.ys) else np.array([0])
        all_v = np.concatenate([v for v in self.vs if len(v)>0]) if any(len(v)>0 for v in self.vs) else np.array([0])
        
        self.x_lim = (np.min(all_x), np.max(all_x))
        self.y_lim = (np.min(all_y), np.max(all_y))
        self.v_lim = (np.min(all_v), np.max(all_v))
        
        all_errs = np.concatenate(list(self.diffs.values()))
        self.global_err_max = (np.percentile(all_errs, 99.5) * 1.2) if len(all_errs)>0 else 10.0

        self.fig = plt.figure(figsize=(14, 10))
        self.fig.subplots_adjust(bottom=0.2)
        self.gs = self.fig.add_gridspec(2, 3)
        self.setup_axes()
        self.create_lines()
        
        self.stat_text = self.fig.text(0.22, 0.015, "", fontsize=10, family='monospace', 
                                       verticalalignment='bottom',
                                       bbox=dict(facecolor='#f0f0f0', alpha=0.9, edgecolor='silver', boxstyle='round,pad=0.5'))

        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Time', 0, self.max_time - self.window_size, valinit=0)
        self.slider.on_changed(self.update_from_slider)

        ax_prev = plt.axes([0.7, 0.02, 0.08, 0.04])
        ax_next = plt.axes([0.79, 0.02, 0.08, 0.04])
        self.b_prev = Button(ax_prev, '<< Prev')
        self.b_next = Button(ax_next, 'Next >>')
        self.b_prev.on_clicked(self.prev_page)
        self.b_next.on_clicked(self.next_page)
        
        ax_snap = plt.axes([0.88, 0.02, 0.08, 0.04])
        self.b_snap = Button(ax_snap, 'Snap 📷')
        self.b_snap.on_clicked(self.save_snapshot)

        ax_check = plt.axes([0.02, 0.02, 0.19, 0.07], frameon=False)
        self.check = CheckButtons(ax_check, ['Run 1', 'Run 2', 'Run 3'], [True, True, True])
        self.style_checkboxes()
        self.check.on_clicked(self.toggle_visibility)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_view()
        plt.show()

    def style_checkboxes(self):
        colors = ['blue', 'red', 'green']
        try:
            for i, rect in enumerate(self.check.rectangles):
                rect.set_facecolor(colors[i])
                rect.set_alpha(0.5)
                rect.set_width(0.15) 
                rect.set_height(0.3) 
                
            for line in self.check.lines:
                line.set_linewidth(2.5)
        except Exception as e:
            pass

    def setup_axes(self):
        self.ax_time = self.fig.add_subplot(self.gs[0, :])
        self.ax_time.set_title("Time Domain")
        self.ax_time.set_ylabel("Position X")
        self.ax_time.grid(True)

        self.ax_traj = self.fig.add_subplot(self.gs[1, 0])
        self.ax_traj.set_title("Trajectory (X vs Y)")
        pad_x = (self.x_lim[1]-self.x_lim[0])*0.1 if (self.x_lim[1]-self.x_lim[0])>0 else 10
        pad_y = (self.y_lim[1]-self.y_lim[0])*0.1 if (self.y_lim[1]-self.y_lim[0])>0 else 10
        self.ax_traj.set_xlim(self.x_lim[0]-pad_x, self.x_lim[1]+pad_x)
        self.ax_traj.set_ylim(self.y_lim[1]+pad_y, self.y_lim[0]-pad_y) # Inverted Y
        self.ax_traj.grid(True)

        self.ax_err = self.fig.add_subplot(self.gs[1, 1])
        self.ax_err.set_title("Deviations")
        self.ax_err.grid(True)
        self.ax_err.set_ylim(0, self.global_err_max)

        self.ax_phase = self.fig.add_subplot(self.gs[1, 2])
        self.ax_phase.set_title("Phase Space (X vs V)")
        
        pad_v = (self.v_lim[1]-self.v_lim[0])*0.1 if (self.v_lim[1]-self.v_lim[0])>0 else 0.5
        self.ax_phase.set_ylim(self.v_lim[0]-pad_v, self.v_lim[1]+pad_v)
        self.ax_phase.set_xlim(self.x_lim[0]-pad_x, self.x_lim[1]+pad_x)
        self.ax_phase.grid(True)

    def create_lines(self):
        self.lines_time = [
            self.ax_time.plot([], [], 'b-', alpha=0.8, lw=1.5)[0],
            self.ax_time.plot([], [], 'r--', alpha=0.8, lw=1.5)[0],
            self.ax_time.plot([], [], 'g-.', alpha=0.8, lw=1.5)[0]
        ]
        self.lines_traj = [
            self.ax_traj.plot([], [], 'b-', alpha=0.6, lw=1)[0],
            self.ax_traj.plot([], [], 'r--', alpha=0.6, lw=1)[0],
            self.ax_traj.plot([], [], 'g-.', alpha=0.6, lw=1)[0]
        ]
        self.lines_phase = [
            self.ax_phase.plot([], [], 'b-', alpha=0.5, lw=1)[0],
            self.ax_phase.plot([], [], 'r--', alpha=0.5, lw=1)[0],
            self.ax_phase.plot([], [], 'g-.', alpha=0.5, lw=1)[0]
        ]
        self.line_err12 = self.ax_err.plot([], [], color='purple', lw=1.2, label="|R1-R2|")[0]
        self.line_err13 = self.ax_err.plot([], [], color='orange', lw=1.2, label="|R1-R3|")[0]
        self.line_err23 = self.ax_err.plot([], [], color='#008080', lw=1.5, ls=':', label="|R2-R3|")[0]
        self.ax_err.legend(loc="upper right", fontsize='small')

    def toggle_visibility(self, label):
        idx = 0 if 'Run 1' in label else (1 if 'Run 2' in label else 2)
        self.visible[idx] = not self.visible[idx]
        self.lines_time[idx].set_visible(self.visible[idx])
        self.lines_traj[idx].set_visible(self.visible[idx])
        self.lines_phase[idx].set_visible(self.visible[idx])
        self.line_err12.set_visible(self.visible[0] and self.visible[1])
        self.line_err13.set_visible(self.visible[0] and self.visible[2])
        self.line_err23.set_visible(self.visible[1] and self.visible[2])
        
        # FIX: Immediately update statistics text when checkbox changes
        self.update_stats_text(self.current_start, self.current_start + self.window_size)
        self.fig.canvas.draw_idle()

    def update_stats_text(self, start, end):
        # 1. Collect strings for Damping and Period only for visible runs
        damp_strs = []
        period_strs = []
        
        for i in range(3):
            if not self.visible[i]: continue # Skip if hidden
            
            # Damping
            val = self.dampings[i][0]
            damp_strs.append(f"R{i+1}={val:.5f}")
            
            # Period
            active, _ = get_local_activity(self.ts[i], self.xs[i], start, end)
            T, c = calculate_robust_period(self.ts[i], self.peaks[i], start, end) if active else (None, 0)
            p_val = f"{T:.4f}s ({c})" if T else "-"
            period_strs.append(f"R{i+1}={p_val}")

        # 2. Collect RMS only if BOTH compared runs are visible
        rms_strs = []
        mask_d = (self.t_common >= start) & (self.t_common < end)
        
        def get_rms(arr):
            seg = arr[mask_d]
            return np.sqrt(np.mean(seg**2)) if len(seg)>0 else 0.0

        # Define pairs and their dict keys: (RunA, RunB, Key)
        # Indicies are 0-based, Keys are '12', '13', '23'
        pairs = [(0, 1, '12'), (0, 2, '13'), (1, 2, '23')]
        
        for i, j, key in pairs:
            if self.visible[i] and self.visible[j]:
                val = get_rms(self.diffs[key])
                rms_strs.append(f"|{i+1}-{j+1}|={val:.2f}")

        # 3. Construct final lines
        # Use comma join, or a placeholder if empty to keep layout stable-ish
        line_d = "DAMPING: " + (", ".join(damp_strs) if damp_strs else "")
        line_p = "PERIOD:  " + (", ".join(period_strs) if period_strs else "")
        line_r = "RMS ERR: " + (", ".join(rms_strs) if rms_strs else "")

        text = f"{line_d}\n{line_p}\n{line_r}"
        self.stat_text.set_text(text)

    def update_from_slider(self, val):
        self.current_start = val
        self.update_view()

    def update_view(self):
        start, end = self.current_start, self.current_start + self.window_size
        self.ax_time.set_xlim(start, end); self.ax_err.set_xlim(start, end)
        
        y_vals = []
        for i in range(3):
            if not self.visible[i] or len(self.ts[i]) == 0: continue
            mask = (self.ts[i] >= start) & (self.ts[i] < end)
            t_seg, x_seg = self.ts[i][mask], self.xs[i][mask]
            self.lines_time[i].set_data(t_seg, x_seg)
            self.lines_traj[i].set_data(x_seg, self.ys[i][mask])
            self.lines_phase[i].set_data(x_seg, self.vs[i][mask])
            if len(x_seg)>0: y_vals.append(x_seg)

        if y_vals:
            all_v = np.concatenate(y_vals)
            mn, mx = np.min(all_v), np.max(all_v)
            pad = (mx-mn)*0.1 if (mx-mn)>0 else 5
            self.ax_time.set_ylim(mn-pad, mx+pad)

        mask_d = (self.t_common >= start) & (self.t_common < end)
        t_d = self.t_common[mask_d]
        self.line_err12.set_data(t_d, self.diffs['12'][mask_d])
        self.line_err13.set_data(t_d, self.diffs['13'][mask_d])
        self.line_err23.set_data(t_d, self.diffs['23'][mask_d])

        self.fig.suptitle(f"Pendulum V21 | T: {start:.1f}s", fontsize=12)
        self.update_stats_text(start, end)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = self.window_size * 0.25
        if event.key == 'right': self.current_start = min(self.current_start + step, self.max_time - self.window_size)
        elif event.key == 'left': self.current_start = max(self.current_start - step, 0)
        elif event.key == 'down': self.window_size = min(self.window_size * 1.5, self.max_time)
        elif event.key == 'up': self.window_size = max(self.window_size * 0.66, 2.0)
        self.slider.eventson = False; self.slider.set_val(self.current_start); self.slider.eventson = True
        self.update_view()

    def next_page(self, event):
        self.current_start = min(self.current_start + self.window_size, self.max_time - self.window_size)
        self.slider.set_val(self.current_start)

    def prev_page(self, event):
        self.current_start = max(self.current_start - self.window_size, 0)
        self.slider.set_val(self.current_start)
        
    def save_snapshot(self, event):
        fname = f"snapshot_{int(self.current_start)}.png"
        self.fig.savefig(fname)
        print(f"Snapshot saved: {fname}")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="CSV files")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    files = [FILE_1_DEFAULT, FILE_2_DEFAULT, FILE_3_DEFAULT]
    if args.files:
        for i in range(min(len(args.files), 3)): files[i] = args.files[i]

    print(f"Files: {files}")
    data = [load_data(f) for f in files]
    ts, xs, ys = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    
    if len(ts[0]) == 0: print("Error: Run 1 empty"); return
    for i in range(3):
        if len(ts[i])>0: ts[i] -= ts[i][0]

    if AUTO_SYNC_PHASE and not args.no_sync:
        print("Syncing by FIRST PEAK...")
        if len(ts[1])>0: ts[1] -= sync_by_first_peak(ts[0], xs[0], ts[1], xs[1])
        if len(ts[2])>0: ts[2] -= sync_by_first_peak(ts[0], xs[0], ts[2], xs[2])

    TimeNavigator(ts, xs, ys, files)

if __name__ == "__main__":
    run()