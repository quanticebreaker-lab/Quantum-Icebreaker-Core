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
    page_title="Pendulum Analysis V20.Web",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ROBUST MATH UTILS (COPIED FROM V20.4)
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

def detrend_linear(sig):
    if len(sig) < 2: return sig
    t = np.arange(len(sig))
    p = np.polyfit(t, sig, 1)
    return sig - (p[0]*t + p[1])

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
    
    x1_u, x2_u = detrend_linear(x1_u), detrend_linear(x2_u)
    x1_u, x2_u = (x1_u - np.mean(x1_u)), (x2_u - np.mean(x2_u))
    s1, s2 = np.std(x1_u), np.std(x2_u)
    if s1 == 0 or s2 == 0: return 0.0
    
    x1_u /= s1; x2_u /= s2
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

def calculate_robust_period(t, peaks_idx, view_start, view_end, min_cycles=5):
    if len(peaks_idx) < 2: return None, 0
    t_peaks = t[peaks_idx]
    mask = (t_peaks >= view_start) & (t_peaks <= view_end)
    sel = t_peaks[mask]
    if len(sel) < 2:
        idx_closest = (np.abs(t_peaks - (view_start+view_end)/2.0)).argmin()
        sel = t_peaks[max(0, idx_closest - min_cycles//2) : min(len(t_peaks), idx_closest + min_cycles)]
    return (np.mean(np.diff(sel)), len(sel)) if len(sel) >= 2 else (None, 0)

def get_local_activity(t, x, start, end, threshold):
    mask = (t >= start) & (t < end)
    if not np.any(mask): return False, 0.0
    p2p = np.max(x[mask]) - np.min(x[mask])
    return p2p > threshold, p2p

# ==========================================
# FILE LOADING (ADAPTED FOR WEB)
# ==========================================
def smart_find_column(header, candidates):
    header_lower = [h.lower().strip() for h in header]
    for candidate in candidates:
        for i, col_name in enumerate(header_lower):
            if candidate in col_name: return i
    return None

def parse_uploaded_file(uploaded_file):
    """Parses a Streamlit UploadedFile object."""
    times, xs, ys = [], [], []
    if uploaded_file is None:
        return np.array([]), np.array([]), np.array([])
    
    try:
        # Convert bytes to string buffer
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
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return np.array([]), np.array([]), np.array([])

# ==========================================
# MAIN PROCESSING (CACHED)
# ==========================================
@st.cache_data
def process_data(uploaded_files, auto_sync, resample_dt, polyorder, median_kernel, use_median):
    """
    Heavy lifting: Loading, Syncing, Filtering.
    Runs once when files change.
    """
    raw_data = []
    
    # Ensure we always have 3 slots, even if empty
    for i in range(3):
        if i < len(uploaded_files):
            t, x, y = parse_uploaded_file(uploaded_files[i])
            if len(t) > 0: t -= t[0] # Zero start
            raw_data.append({'t': t, 'x': x, 'y': y, 'name': uploaded_files[i].name})
        else:
            raw_data.append({'t': np.array([]), 'x': np.array([]), 'y': np.array([]), 'name': f"Run {i+1}"})

    # 1. Sync
    if auto_sync and len(raw_data[0]['t']) > 0:
        ref_t, ref_x = raw_data[0]['t'], raw_data[0]['x']
        for i in [1, 2]:
            if len(raw_data[i]['t']) > 0:
                lag = sync_signals_fft(ref_t, ref_x, raw_data[i]['t'], raw_data[i]['x'], resample_dt)
                raw_data[i]['t'] -= lag

    # 2. Filter & Calculate Velocities/Peaks
    processed = []
    max_t = 0.0
    
    for i, d in enumerate(raw_data):
        t, x, y = d['t'], d['x'], d['y']
        
        if len(t) == 0:
            processed.append({
                't': t, 'x': x, 'y': y, 'v': np.array([]), 
                'peaks': np.array([]), 'damping': (0.0, 0),
                'active': False
            })
            continue
            
        max_t = max(max_t, t[-1])
        dt = get_sampling_rate(t)
        win = calculate_adaptive_window(dt, polyorder)
        
        # Median Filter
        if use_median:
            k = make_odd(median_kernel)
            if len(x) > k: x, y = medfilt(x, k), medfilt(y, k)
        
        # SavGol Filter
        xf = safe_savgol(x, win, polyorder)
        yf = safe_savgol(y, win, polyorder)
        
        # Velocity
        if len(xf) > win:
            vf = savgol_filter(xf, win, polyorder, deriv=1, delta=dt)
        else:
            vf = np.zeros_like(xf)
            
        # Peaks & Damping
        pks = pre_calculate_peaks(xf)
        damp = calculate_damping_robust(t, xf, pks, max_time=None)
        
        processed.append({
            't': t, 'x': xf, 'y': yf, 'v': vf, 
            'peaks': pks, 'damping': damp,
            'active': True
        })

    # 3. Calculate Diffs (Interpolation)
    t_common = np.arange(0, max_t, resample_dt)
    xi = []
    for d in processed:
        xi.append(safe_interp(d['t'], d['x'], t_common))
    
    diffs = {
        '12': np.abs(xi[0]-xi[1]), 
        '13': np.abs(xi[0]-xi[2]), 
        '23': np.abs(xi[1]-xi[2])
    }
    
    # Smooth diffs
    for k in diffs:
        if len(diffs[k]) > 31: 
            diffs[k] = savgol_filter(diffs[k], 31, 3)

    return processed, diffs, t_common, max_t

# ==========================================
# UI & PLOTTING
# ==========================================
def main():
    st.sidebar.title("🔧 Settings")
    
    # --- Sidebar: Controls ---
    uploaded_files = st.sidebar.file_uploader("Upload CSV Files (Max 3)", accept_multiple_files=True, type=['csv'])
    
    st.sidebar.subheader("Parameters")
    auto_sync = st.sidebar.checkbox("Auto-Sync Phase", value=True)
    window_size = st.sidebar.number_input("View Window (s)", value=20.0, min_value=1.0, step=1.0)
    
    with st.sidebar.expander("Advanced Filtering"):
        resample_dt = st.number_input("Resample DT", value=0.033, format="%.3f")
        sg_poly = st.number_input("SavGol PolyOrder", value=3, min_value=1)
        use_median = st.checkbox("Median Prefilter", value=True)
        stop_thresh = st.number_input("Stop Threshold (px)", value=3.0)

    st.sidebar.subheader("Visibility")
    show_r1 = st.sidebar.checkbox("Run 1 (Blue)", value=True)
    show_r2 = st.sidebar.checkbox("Run 2 (Red)", value=True)
    show_r3 = st.sidebar.checkbox("Run 3 (Green)", value=True)
    visible_flags = [show_r1, show_r2, show_r3]
    colors = ['b', 'r', 'g']
    styles = ['-', '--', '-.']

    # --- Main Area ---
    st.title("Pendulum Analysis Dashboard V20.Web")

    if not uploaded_files:
        st.info("👋 Please upload CSV files in the sidebar to start analysis.")
        return

    # Process Data
    processed, diffs, t_common, max_time = process_data(
        uploaded_files, auto_sync, resample_dt, sg_poly, 5, use_median
    )

    # --- Time Slider ---
    start_time = st.slider("Time Navigator (seconds)", 0.0, max(0.0, max_time - window_size), 0.0, step=0.5)
    end_time = start_time + window_size
    
    # --- Statistics Calculation (Dynamic) ---
    st.markdown("### 📊 Statistics & Errors")
    
    # Columns for stats
    c1, c2, c3 = st.columns(3)
    
    stats_data = []
    
    # 1. Damping & Period Stats
    for i in range(3):
        if not visible_flags[i] or not processed[i]['active']: continue
        
        p = processed[i]
        # Check local activity
        is_moving, _ = get_local_activity(p['t'], p['x'], start_time, end_time, stop_thresh)
        
        # Calculate Period
        T, c = (None, 0)
        if is_moving:
            T, c = calculate_robust_period(p['t'], p['peaks'], start_time, end_time)
        
        damp_val = p['damping'][0]
        per_str = f"{T:.4f}s" if T else "-"
        
        stats_data.append({
            "Run": f"Run {i+1}",
            "Damping": f"{damp_val:.5f}",
            "Period (Local)": per_str,
            "Cycles": c
        })

    # 2. RMS Errors
    rms_data = {}
    mask_common = (t_common >= start_time) & (t_common < end_time)
    
    pairs = [(0, 1, '12'), (0, 2, '13'), (1, 2, '23')]
    for i, j, k in pairs:
        if visible_flags[i] and visible_flags[j] and processed[i]['active'] and processed[j]['active']:
            seg = diffs[k][mask_common]
            val = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0.0
            rms_data[f"|R{i+1}-R{j+1}|"] = f"{val:.2f}"

    # Display Stats
    with c1:
        st.markdown("**Single Run Stats**")
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
    
    with c2:
        st.markdown("**RMS Deviation (Window)**")
        if rms_data:
            st.json(rms_data)
        else:
            st.caption("Enable multiple runs to see deviation.")

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3)
    
    # Axes
    ax_time = fig.add_subplot(gs[0, :])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])
    ax_phase = fig.add_subplot(gs[1, 2])
    
    # Limits logic
    all_x, all_y, all_v = [], [], []
    for i in range(3):
        if visible_flags[i] and processed[i]['active']:
            all_x.append(processed[i]['x'])
            all_y.append(processed[i]['y'])
            all_v.append(processed[i]['v'])
            
    if all_x:
        flat_x = np.concatenate(all_x)
        flat_y = np.concatenate(all_y)
        flat_v = np.concatenate(all_v)
        
        xlim = (np.min(flat_x), np.max(flat_x))
        ylim = (np.min(flat_y), np.max(flat_y))
        vlim = (np.min(flat_v), np.max(flat_v))
        
        pad_x = (xlim[1]-xlim[0])*0.1 if (xlim[1]-xlim[0])>0 else 10
        pad_y = (ylim[1]-ylim[0])*0.1 if (ylim[1]-ylim[0])>0 else 10
        pad_v = (vlim[1]-vlim[0])*0.1 if (vlim[1]-vlim[0])>0 else 0.5
        
        ax_traj.set_xlim(xlim[0]-pad_x, xlim[1]+pad_x)
        ax_traj.set_ylim(ylim[1]+pad_y, ylim[0]-pad_y) # Inverted Y
        
        ax_phase.set_xlim(xlim[0]-pad_x, xlim[1]+pad_x)
        ax_phase.set_ylim(vlim[0]-pad_v, vlim[1]+pad_v)

    # Plot Loop
    for i in range(3):
        if not visible_flags[i] or not processed[i]['active']: continue
        
        p = processed[i]
        mask = (p['t'] >= start_time) & (p['t'] < end_time)
        
        t_seg = p['t'][mask]
        x_seg = p['x'][mask]
        y_seg = p['y'][mask]
        v_seg = p['v'][mask]
        
        if len(t_seg) > 0:
            ax_time.plot(t_seg, x_seg, color=colors[i], ls=styles[i], alpha=0.8, label=f"Run {i+1}")
            ax_traj.plot(x_seg, y_seg, color=colors[i], ls=styles[i], alpha=0.6)
            ax_phase.plot(x_seg, v_seg, color=colors[i], ls=styles[i], alpha=0.5)

    # Plot Errors
    t_d = t_common[mask_common]
    if visible_flags[0] and visible_flags[1]:
        ax_err.plot(t_d, diffs['12'][mask_common], color='purple', label="|R1-R2|")
    if visible_flags[0] and visible_flags[2]:
        ax_err.plot(t_d, diffs['13'][mask_common], color='orange', label="|R1-R3|")
    if visible_flags[1] and visible_flags[2]:
        ax_err.plot(t_d, diffs['23'][mask_common], color='teal', ls=':', label="|R2-R3|")

    # Titles & Formatting
    ax_time.set_title("Time Domain")
    ax_time.set_ylabel("Position X")
    ax_time.set_xlim(start_time, end_time)
    ax_time.legend(loc='upper right')
    ax_time.grid(True)
    
    ax_traj.set_title("Trajectory (X vs Y)")
    ax_traj.grid(True)
    
    ax_err.set_title("Deviations")
    ax_err.set_xlim(start_time, end_time)
    ax_err.grid(True)
    ax_err.legend()
    
    ax_phase.set_title("Phase Space (X vs V)")
    ax_phase.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()