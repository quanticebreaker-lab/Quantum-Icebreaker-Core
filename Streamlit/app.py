import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter, find_peaks
import io
import csv
import os

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pendulum V27 Stable",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ИМЕНА ФАЙЛОВ НА GITHUB
DEFAULT_FILES = ["run1.csv", "run2.csv", "run3.csv"] 

# --- CSS ---
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem !important;
                    padding-bottom: 0rem !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                    margin-top: -20px !important;
                }
                h1 { font-size: 1.2rem; margin-bottom: 0.2rem; }
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                
                div.stButton > button {
                    height: 100%;
                    padding-top: 10px;
                    padding-bottom: 10px;
                    margin-top: 10px; 
                }
        </style>
        """, unsafe_allow_html=True)

# ==========================================
# HELPER CLASS
# ==========================================
class LocalFile:
    def __init__(self, filename):
        self.name = filename
        self.path = filename
    def getvalue(self):
        if not os.path.exists(self.path): return None
        with open(self.path, 'rb') as f: return f.read()

# ==========================================
# MATH UTILS
# ==========================================
def make_odd(n):
    n = int(n)
    if n < 1: n = 1
    if n % 2 == 0: n += 1
    return n

def safe_savgol(y, window, polyorder, dt=None):
    window = make_odd(window)
    if len(y) < window:
        window = make_odd(len(y))
        if window <= polyorder: window = make_odd(polyorder + 1)
    if window <= polyorder: return y 
    try:
        if dt: return savgol_filter(y, window, polyorder, deriv=0, delta=dt)
        return savgol_filter(y, window, polyorder, deriv=0)
    except: return y

def calculate_adaptive_window(dt, sg_polyorder):
    window = int(round(0.7 / dt))
    window = make_odd(window)
    if window < 5: window = 5
    if window <= sg_polyorder: window = make_odd(sg_polyorder + 1)
    return window

def safe_interp(t, x, t_target):
    if len(t) == 0: return np.full_like(t_target, np.nan, dtype=float)
    order = np.argsort(t)
    t_sorted, x_sorted = t[order], x[order]
    mask = ~np.isnan(x_sorted)
    if np.sum(mask) < 2: return np.full_like(t_target, np.nan, dtype=float)
    return np.interp(t_target, t_sorted[mask], x_sorted[mask], left=np.nan, right=np.nan)

def get_sampling_rate(t):
    if len(t) < 2: return 0.033
    dt = np.median(np.diff(t))
    return dt if dt > 0 else 0.033

# --- SYNC ---
def get_first_major_peak_time(t, x):
    if len(x) < 5: return None
    x_clean = x.copy()
    mask = ~np.isnan(x_clean)
    if np.sum(mask) < 5: return None
    x_valid = x_clean[mask]
    t_valid = t[mask]
    x_centered = x_valid - np.mean(x_valid)
    max_amp = np.max(np.abs(x_centered))
    if max_amp == 0: return t_valid[0]
    peaks, _ = find_peaks(x_centered, height=max_amp * 0.4, distance=10)
    if len(peaks) > 0: return t_valid[peaks[0]]
    return t_valid[0]

def sync_by_first_peak_robust(t1, x1, t2, x2):
    t_p1 = get_first_major_peak_time(t1, x1)
    t_p2 = get_first_major_peak_time(t2, x2)
    if t_p1 is not None and t_p2 is not None:
        return t_p2 - t_p1
    return 0.0

# --- ANALYSIS ---
def pre_calculate_peaks(x):
    if len(x) == 0: return np.array([], dtype=int)
    mask = ~np.isnan(x)
    if np.sum(mask) < 2: return np.array([], dtype=int)
    x_c = np.nan_to_num(x, nan=np.nanmean(x))
    x_c = x_c - np.mean(x_c)
    return find_peaks(x_c, prominence=max(1e-6, np.std(x_c) * 0.2))[0]

def calculate_damping_robust(t, x, peaks_idx, max_time=None):
    if len(peaks_idx) < 2: return 0.0, 0
    x_c = x - np.nanmean(x)
    t_peaks, x_peaks = t[peaks_idx], np.abs(x_c[peaks_idx])
    if max_time:
        mask = t_peaks <= max_time
        t_peaks, x_peaks = t_peaks[mask], x_peaks[mask]
    mask_valid = (x_peaks > 1e-6) & (~np.isnan(x_peaks))
    t_fit, amp_fit = t_peaks[mask_valid], x_peaks[mask_valid]
    if len(t_fit) < 3: return 0.0, len(t_fit)
    try:
        slope, _ = np.polyfit(t_fit, np.log(amp_fit), 1)
        return -slope, len(t_fit)
    except: return 0.0, len(t_fit)

def calculate_robust_period(t, peaks_idx, start, end):
    t_peaks = t[peaks_idx]
    sel = t_peaks[(t_peaks >= start) & (t_peaks <= end)]
    if len(sel) < 2: return None, 0
    return np.mean(np.diff(sel)), len(sel)

def get_local_activity(t, x, start, end, threshold):
    mask = (t >= start) & (t < end)
    if not np.any(mask): return False, 0.0
    seg = x[mask]
    if np.all(np.isnan(seg)): return False, 0.0
    return (np.nanmax(seg) - np.nanmin(seg)) > threshold, 0.0

# ==========================================
# FILE PARSING
# ==========================================
def smart_find_column(header, candidates):
    header_lower = [h.lower().strip() for h in header]
    for candidate in candidates:
        for i, col_name in enumerate(header_lower):
            if candidate in col_name: return i
    return None

def parse_file_obj(file_obj):
    times, xs, ys = [], [], []
    if file_obj is None: return np.array([]), np.array([]), np.array([])
    try:
        data_bytes = file_obj.getvalue()
        if data_bytes is None: return np.array([]), np.array([]), np.array([])
        content = data_bytes.decode('utf-8')
        f = io.StringIO(content)
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
            except: continue
        t_arr = np.array(times)
        if len(t_arr) > 0:
            sort_idx = np.argsort(t_arr)
            return t_arr[sort_idx], np.array(xs)[sort_idx], np.array(ys)[sort_idx]
        return np.array([]), np.array([]), np.array([])
    except: return np.array([]), np.array([]), np.array([])

@st.cache_data
def process_data(file_objects, auto_sync, resample_dt, poly, med_k, use_med):
    raw = []
    for i in range(3):
        if i < len(file_objects):
            t, x, y = parse_file_obj(file_objects[i])
            if len(t) > 0: t -= t[0]
            raw.append({'t': t, 'x': x, 'y': y, 'n': file_objects[i].name})
        else:
            raw.append({'t': [], 'x': [], 'y': [], 'n': f"Run {i+1}"})

    if auto_sync and len(raw[0]['t']) > 0:
        ref_t, ref_x = raw[0]['t'], raw[0]['x']
        for i in [1, 2]:
            if len(raw[i]['t']) > 0:
                lag = sync_by_first_peak_robust(ref_t, ref_x, raw[i]['t'], raw[i]['x'])
                raw[i]['t'] -= lag

    processed = []
    max_t = 0.0
    for i, d in enumerate(raw):
        t, x, y = d['t'], d['x'], d['y']
        if len(t) == 0:
            processed.append({'t': t, 'x': x, 'y': y, 'v': [], 'peaks': [], 'damping': (0.0, 0), 'active': False})
            continue
        max_t = max(max_t, t[-1])
        dt = get_sampling_rate(t)
        win = calculate_adaptive_window(dt, poly)
        
        if use_med and len(x) > med_k: x, y = medfilt(x, med_k), medfilt(y, med_k)
        xf, yf = safe_savgol(x, win, poly), safe_savgol(y, win, poly)
        vf = savgol_filter(xf, win, poly, deriv=1, delta=dt) if len(xf) > win else np.zeros_like(xf)
        pks = pre_calculate_peaks(xf)
        damp = calculate_damping_robust(t, xf, pks)
        processed.append({'t': t, 'x': xf, 'y': yf, 'v': vf, 'peaks': pks, 'damping': damp, 'active': True})

    t_com = np.arange(0, max_t, resample_dt)
    xi = [safe_interp(p['t'], p['x'], t_com) for p in processed]
    diffs = {'12': np.abs(xi[0]-xi[1]), '13': np.abs(xi[0]-xi[2]), '23': np.abs(xi[1]-xi[2])}
    for k in diffs:
        if len(diffs[k]) > 31 and not np.isnan(diffs[k]).any():
            diffs[k] = savgol_filter(diffs[k], 31, 3)
    return processed, diffs, t_com, max_t

def main():
    st.sidebar.title("Settings")
    
    # --- AUTO-LOAD LOGIC ---
    uploaded_files = st.sidebar.file_uploader("Upload CSV", accept_multiple_files=True, type=['csv'])
    
    files_to_process = []
    if uploaded_files:
        files_to_process = uploaded_files
        st.sidebar.success(f"Using {len(uploaded_files)} uploads")
    else:
        # Check defaults
        for fname in DEFAULT_FILES:
            if os.path.exists(fname):
                files_to_process.append(LocalFile(fname))
        
        if files_to_process:
            st.sidebar.info(f"Loaded {len(files_to_process)} default files")
        else:
            st.sidebar.warning("No files found. Please upload or check GitHub.")
            st.info(f"Expected: {DEFAULT_FILES}")
            return

    st.sidebar.caption("Controls")
    auto_sync = st.sidebar.checkbox("Smart Sync", value=True)
    window_size = st.sidebar.number_input("Window (s)", value=20.0, min_value=1.0)
    
    with st.sidebar.expander("Filter Params"):
        resample_dt = st.number_input("DT", value=0.033)
        sg_poly = st.number_input("Poly", value=3)
        use_median = st.checkbox("Median", value=True)
        stop_thresh = st.number_input("Stop Thresh", value=3.0)

    vis = [st.sidebar.checkbox(f"R{i+1}", True) for i in range(3)]
    colors, styles = ['b', 'r', 'g'], ['-', '--', '-.']

    st.markdown("""<h3 style='text-align: left; font-size: 20px; margin-top: -10px; margin-bottom: 10px;'>
        Pendulum Analysis V27 (Stable)</h3>""", unsafe_allow_html=True)
    
    processed, diffs, t_common, max_time = process_data(files_to_process, auto_sync, 0.033, 3, 5, True)

    plot_cont = st.container()
    ctrl_cont = st.container()

    # --- CONTROLS ---
    with ctrl_cont:
        if 'start_time' not in st.session_state: st.session_state.start_time = 0.0
        def move_slider(delta):
            new_val = st.session_state.start_time + delta
            max_val = max(0.0, max_time - window_size)
            st.session_state.start_time = max(0.0, min(max_val, new_val))

        c_prev, c_slider, c_next, c_stats = st.columns([1, 15, 1, 8])
        with c_prev: st.button("⏪", on_click=move_slider, args=(-window_size/2,))
        with c_next: st.button("⏩", on_click=move_slider, args=(window_size/2,))
        with c_slider:
            start = st.slider("Time Navigator", 0.0, max(0.0, max_time - window_size), 
                              key='start_time', step=0.5, label_visibility="collapsed")
            end = start + window_size
            
        with c_stats:
            stats_txt = []
            for i in range(3):
                if vis[i] and processed[i]['active']:
                    p = processed[i]
                    mv, _ = get_local_activity(p['t'], p['x'], start, end, 3.0)
                    T, c = calculate_robust_period(p['t'], p['peaks'], start, end) if mv else (None, 0)
                    if T: stats_txt.append(f"**R{i+1}**: Damp={p['damping'][0]:.4f}, T={T:.3f}s")
            
            rms_txt = []
            m_c = (t_common >= start) & (t_common < end)
            for i, j, k in [(0,1,'12'),(0,2,'13'),(1,2,'23')]:
                if vis[i] and vis[j] and processed[i]['active'] and processed[j]['active']:
                    seg = diffs[k][m_c]
                    seg = seg[~np.isnan(seg)]
                    v = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0.0
                    rms_txt.append(f"|{i+1}-{j+1}|={v:.2f}")
            st.markdown(f"<div style='font-size: 12px; margin-top: -5px;'>{' | '.join(stats_txt)}<br>RMS: {', '.join(rms_txt)}</div>", unsafe_allow_html=True)

    # --- PLOTS ---
    with plot_cont:
        fig = plt.figure(figsize=(14, 5.0))
        gs = fig.add_gridspec(2, 4)
        ax_t = fig.add_subplot(gs[0, :])
        ax_tr = fig.add_subplot(gs[1, 0])
        ax_er = fig.add_subplot(gs[1, 1:3])
        ax_ph = fig.add_subplot(gs[1, 3])

        all_v = [p['x'] for i, p in enumerate(processed) if vis[i] and p['active']]
        if all_v:
            fx = np.concatenate(all_v)
            fx = fx[~np.isnan(fx)]
            if len(fx) > 0:
                xlim = (np.min(fx), np.max(fx))
                pad = (xlim[1]-xlim[0])*0.1 if (xlim[1]-xlim[0]) > 0 else 1
                ax_tr.set_xlim(xlim[0]-pad, xlim[1]+pad)
                ax_ph.set_xlim(xlim[0]-pad, xlim[1]+pad)

        for i in range(3):
            if vis[i] and processed[i]['active']:
                p = processed[i]
                m = (p['t'] >= start) & (p['t'] < end)
                if np.any(m):
                    ax_t.plot(p['t'][m], p['x'][m], c=colors[i], ls=styles[i], alpha=0.8, label=f"R{i+1}")
                    ax_tr.plot(p['x'][m], p['y'][m], c=colors[i], ls=styles[i], alpha=0.6)
                    ax_ph.plot(p['x'][m], p['v'][m], c=colors[i], ls=styles[i], alpha=0.5)

        td = t_common[m_c]
        if vis[0] and vis[1]: ax_er.plot(td, diffs['12'][m_c], 'purple', label="|1-2|")
        if vis[0] and vis[2]: ax_er.plot(td, diffs['13'][m_c], 'orange', label="|1-3|")
        if vis[1] and vis[2]: ax_er.plot(td, diffs['23'][m_c], 'teal', ls=':', label="|2-3|")

        ax_t.set_xlim(start, end); ax_t.grid(True); ax_t.legend(loc='upper right', fontsize=8)
        ax_t.set_title(f"Time (T={start:.1f}s)", fontsize=10, pad=2)
        ax_t.tick_params(labelsize=8)
        
        for ax, tit in zip([ax_tr, ax_er, ax_ph], ["Trajectory", "Errors", "Phase"]):
            ax.set_title(tit, fontsize=9, pad=2); ax.grid(True); ax.tick_params(labelsize=7)
        ax_er.set_xlim(start, end)

        plt.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08, hspace=0.3, wspace=0.2)
        st.pyplot(fig)
        
        # --- MEMORY LEAK FIX ---
        plt.close(fig)

if __name__ == "__main__":
    main()