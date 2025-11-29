import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter, find_peaks, fftconvolve
import io
import csv

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pendulum V22",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS: REMOVE PADDING & SHRINK TITLE ---
st.markdown("""
        <style>
               /* Remove top padding completely */
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                /* Make Title Small */
                h1 {
                    font-size: 1.2rem; /* Small font size */
                    margin-bottom: 0.2rem;
                    padding-top: 0.5rem;
                }
                /* Compact text */
                p, label {
                    font-size: 0.9rem;
                }
        </style>
        """, unsafe_allow_html=True)

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
        return y

def calculate_adaptive_window(dt, sg_polyorder):
    target_duration = 0.7
    window = int(round(target_duration / dt))
    window = make_odd(window)
    if window < 5: window = 5
    if window <= sg_polyorder: window = make_odd(sg_polyorder + 1)
    return window

def safe_interp(t, x, t_target):
    if len(t) == 0: return np.zeros_like(t_target)
    order = np.argsort(t)
    t_sorted, x_sorted = t[order], x[order]
    mask = ~np.isnan(x_sorted)
    if np.sum(mask) < 2: return np.full_like(t_target, np.nan, dtype=float)
    return np.interp(t_target, t_sorted[mask], x_sorted[mask])

def get_sampling_rate(t):
    if len(t) < 2: return 0.033
    dt = np.median(np.diff(t))
    return dt if dt > 0 else 0.033

def sync_signals_fft(t1, x1, t2, x2, dt):
    if len(t1) < 10 or len(t2) < 10: return 0.0
    duration = min(t1[-1] - t1[0], t2[-1] - t2[0])
    if duration <= 0: return 0.0
    
    t_uniform = np.arange(0, duration, dt)
    x1_u = safe_interp(t1 - t1[0], x1, t_uniform)
    x2_u = safe_interp(t2 - t2[0], x2, t_uniform)
    
    x1_p = np.polyfit(np.arange(len(x1_u)), x1_u, 1)
    x2_p = np.polyfit(np.arange(len(x2_u)), x2_u, 1)
    x1_u = x1_u - (x1_p[0]*np.arange(len(x1_u)) + x1_p[1])
    x2_u = x2_u - (x2_p[0]*np.arange(len(x2_u)) + x2_p[1])
    
    x1_u = (x1_u - np.mean(x1_u))
    x2_u = (x2_u - np.mean(x2_u))
    
    corr = fftconvolve(x1_u, x2_u[::-1], mode='full')
    lags = np.arange(-len(x2_u) + 1, len(x1_u))
    best_lag = lags[np.argmax(corr)]
    return best_lag * dt

def pre_calculate_peaks(x):
    if len(x) == 0: return np.array([], dtype=int)
    x_c = x - np.mean(x)
    std_val = np.std(x_c)
    if std_val == 0: return np.array([], dtype=int)
    peaks, _ = find_peaks(x_c, prominence=max(1e-6, std_val * 0.2))
    return peaks

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

def calculate_robust_period(t, peaks_idx, view_start, view_end):
    if len(peaks_idx) < 2: return None, 0
    t_peaks = t[peaks_idx]
    mask = (t_peaks >= view_start) & (t_peaks <= view_end)
    sel = t_peaks[mask]
    if len(sel) < 2:
        idx_closest = (np.abs(t_peaks - (view_start+view_end)/2.0)).argmin()
        sel = t_peaks[max(0, idx_closest - 2) : min(len(t_peaks), idx_closest + 3)]
    return (np.mean(np.diff(sel)), len(sel)) if len(sel) >= 2 else (None, 0)

def get_local_activity(t, x, start, end, threshold):
    mask = (t >= start) & (t < end)
    if not np.any(mask): return False, 0.0
    p2p = np.max(x[mask]) - np.min(x[mask])
    return p2p > threshold, p2p

# ==========================================
# FILE LOADING
# ==========================================
def smart_find_column(header, candidates):
    header_lower = [h.lower().strip() for h in header]
    for candidate in candidates:
        for i, col_name in enumerate(header_lower):
            if candidate in col_name: return i
    return None

def parse_uploaded_file(uploaded_file):
    times, xs, ys = [], [], []
    if uploaded_file is None: return np.array([]), np.array([]), np.array([])
    try:
        content = uploaded_file.getvalue().decode('utf-8')
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
            except ValueError: continue
        
        t_arr, x_arr, y_arr = np.array(times), np.array(xs), np.array(ys)
        if len(t_arr) > 0:
            sort_idx = np.argsort(t_arr)
            return t_arr[sort_idx], x_arr[sort_idx], y_arr[sort_idx]
        return np.array([]), np.array([]), np.array([])
    except: return np.array([]), np.array([]), np.array([])

# ==========================================
# MAIN PROCESSING
# ==========================================
@st.cache_data
def process_data(uploaded_files, auto_sync, resample_dt, polyorder, median_kernel, use_median):
    raw_data = []
    for i in range(3):
        if i < len(uploaded_files):
            t, x, y = parse_uploaded_file(uploaded_files[i])
            if len(t) > 0: t -= t[0]
            raw_data.append({'t': t, 'x': x, 'y': y, 'name': uploaded_files[i].name})
        else:
            raw_data.append({'t': np.array([]), 'x': np.array([]), 'y': np.array([]), 'name': f"Run {i+1}"})

    if auto_sync and len(raw_data[0]['t']) > 0:
        ref_t, ref_x = raw_data[0]['t'], raw_data[0]['x']
        for i in [1, 2]:
            if len(raw_data[i]['t']) > 0:
                lag = sync_signals_fft(ref_t, ref_x, raw_data[i]['t'], raw_data[i]['x'], resample_dt)
                raw_data[i]['t'] -= lag

    processed = []
    max_t = 0.0
    
    for i, d in enumerate(raw_data):
        t, x, y = d['t'], d['x'], d['y']
        if len(t) == 0:
            processed.append({'t': t, 'x': x, 'y': y, 'v': np.array([]), 'peaks': np.array([]), 'damping': (0.0, 0), 'active': False})
            continue
        max_t = max(max_t, t[-1])
        dt = get_sampling_rate(t)
        win = calculate_adaptive_window(dt, polyorder)
        if use_median:
            k = make_odd(median_kernel)
            if len(x) > k: x, y = medfilt(x, k), medfilt(y, k)
        xf = safe_savgol(x, win, polyorder)
        yf = safe_savgol(y, win, polyorder)
        vf = savgol_filter(xf, win, polyorder, deriv=1, delta=dt) if len(xf) > win else np.zeros_like(xf)
        pks = pre_calculate_peaks(xf)
        damp = calculate_damping_robust(t, xf, pks, max_time=None)
        processed.append({'t': t, 'x': xf, 'y': yf, 'v': vf, 'peaks': pks, 'damping': damp, 'active': True})

    t_common = np.arange(0, max_t, resample_dt)
    xi = [safe_interp(d['t'], d['x'], t_common) for d in processed]
    diffs = { '12': np.abs(xi[0]-xi[1]), '13': np.abs(xi[0]-xi[2]), '23': np.abs(xi[1]-xi[2]) }
    for k in diffs:
        if len(diffs[k]) > 31: diffs[k] = savgol_filter(diffs[k], 31, 3)

    return processed, diffs, t_common, max_t

# ==========================================
# UI & COMPACT PLOTTING
# ==========================================
def main():
    st.sidebar.title("Settings")
    uploaded_files = st.sidebar.file_uploader("Upload CSV (Max 3)", accept_multiple_files=True, type=['csv'])
    
    st.sidebar.caption("Controls")
    auto_sync = st.sidebar.checkbox("Auto-Sync", value=True)
    window_size = st.sidebar.number_input("Window (s)", value=20.0, min_value=1.0)
    
    with st.sidebar.expander("Filter Params"):
        resample_dt = st.number_input("DT", value=0.033)
        sg_poly = st.number_input("Poly", value=3)
        use_median = st.checkbox("Median", value=True)
        stop_thresh = st.number_input("Stop Thresh", value=3.0)

    st.sidebar.caption("Visibility")
    vis = [st.sidebar.checkbox(f"R{i+1}", True) for i in range(3)]
    colors, styles = ['b', 'r', 'g'], ['-', '--', '-.']

    # --- SMALL TITLE ---
    st.title("Pendulum Analysis Dashboard V20.Web")

    if not uploaded_files:
        st.info("Upload CSVs in sidebar.")
        return

    processed, diffs, t_common, max_time = process_data(uploaded_files, auto_sync, resample_dt, sg_poly, 5, use_median)

    plot_cont = st.container()
    ctrl_cont = st.container()

    with ctrl_cont:
        c_slider, c_stats = st.columns([1, 1])
        with c_slider:
            start_time = st.slider("Time (s)", 0.0, max(0.0, max_time - window_size), 0.0, step=0.5)
            end_time = start_time + window_size
        with c_stats:
            stat_txt = []
            for i in range(3):
                if vis[i] and processed[i]['active']:
                    p = processed[i]
                    is_mv, _ = get_local_activity(p['t'], p['x'], start_time, end_time, stop_thresh)
                    T, c = (calculate_robust_period(p['t'], p['peaks'], start_time, end_time) if is_mv else (None, 0))
                    stat_txt.append(f"**R{i+1}**: Damp={p['damping'][0]:.4f}, T={T:.3f}s" if T else f"**R{i+1}**: Damp={p['damping'][0]:.4f}")
            rms_txt = []
            mask_c = (t_common >= start_time) & (t_common < end_time)
            for i,j,k in [(0,1,'12'),(0,2,'13'),(1,2,'23')]:
                if vis[i] and vis[j] and processed[i]['active'] and processed[j]['active']:
                    v = np.sqrt(np.mean(diffs[k][mask_c]**2)) if np.any(mask_c) else 0.0
                    rms_txt.append(f"|{i+1}-{j+1}|={v:.2f}")
            st.markdown("  \n".join(stat_txt) + "  \nRMS: " + ", ".join(rms_txt))

    with plot_cont:
        # LOW HEIGHT (5.0) for single screen
        fig = plt.figure(figsize=(14, 5.0)) 
        gs = fig.add_gridspec(2, 4)
        ax_time = fig.add_subplot(gs[0, :])
        ax_traj = fig.add_subplot(gs[1, 0])
        ax_err = fig.add_subplot(gs[1, 1:3])
        ax_phase = fig.add_subplot(gs[1, 3])
        
        all_x = [processed[i]['x'] for i in range(3) if vis[i] and processed[i]['active']]
        all_y = [processed[i]['y'] for i in range(3) if vis[i] and processed[i]['active']]
        all_v = [processed[i]['v'] for i in range(3) if vis[i] and processed[i]['active']]
        
        if all_x:
            fx, fy, fv = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_v)
            xlim, ylim, vlim = (np.min(fx), np.max(fx)), (np.min(fy), np.max(fy)), (np.min(fv), np.max(fv))
            px, py, pv = (xlim[1]-xlim[0])*0.1, (ylim[1]-ylim[0])*0.1, (vlim[1]-vlim[0])*0.1
            ax_traj.set_xlim(xlim[0]-px, xlim[1]+px); ax_traj.set_ylim(ylim[1]+py, ylim[0]-py)
            ax_phase.set_xlim(xlim[0]-px, xlim[1]+px); ax_phase.set_ylim(vlim[0]-pv, vlim[1]+pv)

        for i in range(3):
            if not vis[i] or not processed[i]['active']: continue
            p = processed[i]
            m = (p['t'] >= start_time) & (p['t'] < end_time)
            ts, xs, ys, vs = p['t'][m], p['x'][m], p['y'][m], p['v'][m]
            if len(ts)>0:
                ax_time.plot(ts, xs, c=colors[i], ls=styles[i], alpha=0.8, label=f"R{i+1}")
                ax_traj.plot(xs, ys, c=colors[i], ls=styles[i], alpha=0.6)
                ax_phase.plot(xs, vs, c=colors[i], ls=styles[i], alpha=0.5)

        td = t_common[mask_c]
        if vis[0] and vis[1]: ax_err.plot(td, diffs['12'][mask_c], 'purple', label="|1-2|")
        if vis[0] and vis[2]: ax_err.plot(td, diffs['13'][mask_c], 'orange', label="|1-3|")
        if vis[1] and vis[2]: ax_err.plot(td, diffs['23'][mask_c], 'teal', ls=':', label="|2-3|")

        ax_time.set_xlim(start_time, end_time); ax_time.grid(True); ax_time.legend(loc='upper right', fontsize='small', framealpha=0.5)
        ax_time.tick_params(axis='both', which='major', labelsize=8)
        ax_time.set_title(f"Time Domain (T={start_time:.1f}s)", fontsize=10, pad=2)
        for ax, title in zip([ax_traj, ax_err, ax_phase], ["Trajectory", "Errors", "Phase Space"]):
            ax.set_title(title, fontsize=9, pad=2); ax.grid(True); ax.tick_params(labelsize=7)
        ax_err.set_xlim(start_time, end_time)
        
        plt.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08, hspace=0.3, wspace=0.2)
        st.pyplot(fig)

if __name__ == "__main__":
    main()