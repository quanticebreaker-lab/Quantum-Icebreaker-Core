# krystall_experiment_improved.py
# Улучшённая версия "Digital Primordial Broth" с экспериментальной логикой
# Требования: pygame, numpy, pandas
# Запуск: python krystall_experiment_improved.py --participant Андрей

import pygame
import numpy as np
import random
import argparse
import csv
import datetime
import time
import socket
import threading
import os
import sys

# --------------------------
# --- SIMULATION SETTINGS ---
# --------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
NUM_AGENTS = 300
AGENT_RADIUS = 3
AGENT_SPEED = 2.0
CHAOS_FACTOR = 0.5

# Trial / experiment parameters (меняй)
TRIALS = 20                 # количество пар baseline+intention
BASELINE_SECONDS = 20       # длительность baseline периода (сек)
INTENTION_SECONDS = 20      # длительность intention периода (сек)
INTER_TRIAL_PAUSE = 5       # пауза между парами (сек)
FPS = 60

# Logging / output
OUTPUT_DIR = "krystall_results"
CSV_PREFIX = "krystall"

# UDP trigger (опционально) - слушает на этом порту сигнал начать intention
UDP_ENABLE = True
UDP_PORT = 9999
UDP_MSG = b"START_INTENTION"

# Visualization
VISUALIZE = True  # False для фоновой работы (headless)

# --------------------------
# --- COLORS & UI ---
# --------------------------
BG_COLOR = (5, 5, 20)
AGENT_COLOR = (200, 220, 255)
TEXT_COLOR = (255, 255, 255)

# --------------------------
# --- Agent class ---
# --------------------------
class Agent:
    def __init__(self):
        self.pos = np.array([random.uniform(0, SCREEN_WIDTH), random.uniform(0, SCREEN_HEIGHT)], dtype=float)
        angle = random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float) * AGENT_SPEED

    def update(self):
        chaos_vec = np.random.rand(2) * 2 - 1
        self.vel += chaos_vec * CHAOS_FACTOR
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = self.vel / norm * AGENT_SPEED
        else:
            angle = random.uniform(0, 2 * np.pi)
            self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float) * AGENT_SPEED
        self.pos += self.vel

        # bounce
        if self.pos[0] <= AGENT_RADIUS:
            self.pos[0] = AGENT_RADIUS
            self.vel[0] *= -1
        elif self.pos[0] >= SCREEN_WIDTH - AGENT_RADIUS:
            self.pos[0] = SCREEN_WIDTH - AGENT_RADIUS
            self.vel[0] *= -1

        if self.pos[1] <= AGENT_RADIUS:
            self.pos[1] = AGENT_RADIUS
            self.vel[1] *= -1
        elif self.pos[1] >= SCREEN_HEIGHT - AGENT_RADIUS:
            self.pos[1] = SCREEN_HEIGHT - AGENT_RADIUS
            self.vel[1] *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, AGENT_COLOR, self.pos.astype(int), AGENT_RADIUS)

# --------------------------
# --- UDP listener thread ---
# --------------------------
class UDPListener(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.triggered = False

    def run(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data.strip() == UDP_MSG:
                    print(f"[UDP] Trigger received from {addr}")
                    self.triggered = True
            except Exception as e:
                print("[UDP] Listener error:", e)

# --------------------------
# --- Utility: compute GCF ---
# --------------------------
def compute_gcf(agents):
    velocities = np.array([agent.vel for agent in agents])
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_velocities = velocities / norms
    mean_vector = np.mean(normalized_velocities, axis=0)
    return np.linalg.norm(mean_vector)

# --------------------------
# --- Permutation test ---
# --------------------------
def permutation_test(label_array, value_array, n_permutations=5000, seed=0):
    # label_array: 0/1 for baseline/intention
    rng = np.random.default_rng(seed)
    observed_diff = value_array[label_array==1].mean() - value_array[label_array==0].mean()
    count = 0
    all_vals = value_array.copy()
    for _ in range(n_permutations):
        rng.shuffle(label_array)
        diff = all_vals[label_array==1].mean() - all_vals[label_array==0].mean()
        if abs(diff) >= abs(observed_diff):
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return observed_diff, pval

# --------------------------
# --- Main experiment ---
# --------------------------
def run_experiment(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestr = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = os.path.join(OUTPUT_DIR, f"{CSV_PREFIX}_{args.participant}_{timestr}.csv")

    # init pygame
    if VISUALIZE:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Krystall Experiment")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 28)
    else:
        pygame.init()
        screen = None
        clock = None
        font = None

    # create agents
    agents = [Agent() for _ in range(NUM_AGENTS)]

    # UDP
    udp_listener = None
    if UDP_ENABLE:
        udp_listener = UDPListener(UDP_PORT)
        udp_listener.start()
        print(f"[INFO] UDP listener started on port {UDP_PORT}. Send {UDP_MSG!r} to trigger intention.")

    # CSV header
    header = ["utc_ts","local_ts","frame","trial_idx","period","participant","gcf","external_trigger"]
    csvfile = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    writer.writerow(header)

    frame = 0
    data_values = []
    data_labels = []

    try:
        for trial_idx in range(TRIALS):
            # --- baseline ---
            period = "baseline"
            baseline_start = time.time()
            end_time = baseline_start + BASELINE_SECONDS
            while time.time() < end_time:
                # events
                for event in pygame.event.get() if VISUALIZE else []:
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt()

                for agent in agents:
                    agent.update()
                gcf = compute_gcf(agents)

                utc_ts = datetime.datetime.utcnow().isoformat()
                local_ts = datetime.datetime.now().isoformat()
                writer.writerow([utc_ts, local_ts, frame, trial_idx, period, args.participant, f"{gcf:.6f}", udp_listener.triggered if udp_listener else False])
                data_values.append(gcf)
                data_labels.append(0)  # baseline=0

                if VISUALIZE:
                    screen.fill(BG_COLOR)
                    for a in agents: a.draw(screen)
                    txt = font.render(f"Trial {trial_idx+1}/{TRIALS} — BASELINE ({int(end_time-time.time())}s)", True, TEXT_COLOR)
                    screen.blit(txt, (10,10))
                    gcf_text = font.render(f"GCF: {gcf:.4f}", True, TEXT_COLOR)
                    screen.blit(gcf_text, (10,40))
                    pygame.display.flip()
                    clock.tick(FPS)
                frame += 1

            # optional short pause
            pause_until = time.time() + INTER_TRIAL_PAUSE
            while time.time() < pause_until:
                if VISUALIZE:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt()
                    pygame.time.wait(50)
                else:
                    time.sleep(0.05)

            # --- intention ---
            period = "intention"
            intention_start = time.time()
            # wait for UDP trigger if enabled (with timeout of 10s)
            if UDP_ENABLE and udp_listener:
                wait_start = time.time()
                # if not triggered within 10s, we proceed anyway (so experiment doesn't hang)
                while not udp_listener.triggered and time.time() - wait_start < 10:
                    time.sleep(0.05)
                # consume trigger flag
                external_trigger = udp_listener.triggered
                udp_listener.triggered = False
            else:
                external_trigger = False

            end_time = intention_start + INTENTION_SECONDS
            while time.time() < end_time:
                for event in pygame.event.get() if VISUALIZE else []:
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt()

                # Optionally modulate chaos_factor or apply small global bias to test "intention influence"
                # Here we do NOT change simulation dynamics automatically; humans attempt to influence.
                for agent in agents:
                    agent.update()
                gcf = compute_gcf(agents)

                utc_ts = datetime.datetime.utcnow().isoformat()
                local_ts = datetime.datetime.now().isoformat()
                writer.writerow([utc_ts, local_ts, frame, trial_idx, period, args.participant, f"{gcf:.6f}", external_trigger])
                data_values.append(gcf)
                data_labels.append(1)  # intention=1

                if VISUALIZE:
                    screen.fill(BG_COLOR)
                    for a in agents: a.draw(screen)
                    txt = font.render(f"Trial {trial_idx+1}/{TRIALS} — INTENTION ({int(end_time-time.time())}s)", True, TEXT_COLOR)
                    screen.blit(txt, (10,10))
                    gcf_text = font.render(f"GCF: {gcf:.4f}", True, TEXT_COLOR)
                    screen.blit(gcf_text, (10,40))
                    trig_text = font.render(f"external_trigger: {external_trigger}", True, TEXT_COLOR)
                    screen.blit(trig_text, (10,70))
                    pygame.display.flip()
                    clock.tick(FPS)
                frame += 1

            # reset trigger flag
            if udp_listener:
                udp_listener.triggered = False

        # End of trials
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, finishing and saving data...")
    finally:
        csvfile.close()
        if VISUALIZE:
            pygame.quit()

    # --- post hoc stats ---
    vals = np.array(data_values)
    labels = np.array(data_labels)
    # Remove NaNs (just in case)
    valid = ~np.isnan(vals)
    vals = vals[valid]
    labels = labels[valid]

    if len(vals) == 0:
        print("[WARN] No data collected.")
        return

    mean_baseline = vals[labels==0].mean() if np.any(labels==0) else np.nan
    mean_intention = vals[labels==1].mean() if np.any(labels==1) else np.nan
    observed_diff = mean_intention - mean_baseline
    # permutation test (more iterations can be run if wanted)
    diff, pval = permutation_test(labels.copy(), vals.copy(), n_permutations=5000, seed=42)

    # bootstrap CI for diff
    rng = np.random.default_rng(0)
    boot_diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(vals), len(vals))
        lab = labels[idx]
        v = vals[idx]
        if np.any(lab==1) and np.any(lab==0):
            boot_diffs.append(v[lab==1].mean() - v[lab==0].mean())
    if boot_diffs:
        lower = np.percentile(boot_diffs, 2.5)
        upper = np.percentile(boot_diffs, 97.5)
    else:
        lower, upper = np.nan, np.nan

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"{CSV_PREFIX}_{args.participant}_{timestr}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"participant: {args.participant}\n")
        f.write(f"trials: {TRIALS}\n")
        f.write(f"baseline_seconds: {BASELINE_SECONDS}\n")
        f.write(f"intention_seconds: {INTENTION_SECONDS}\n")
        f.write(f"mean_baseline: {mean_baseline:.6f}\n")
        f.write(f"mean_intention: {mean_intention:.6f}\n")
        f.write(f"observed_diff: {observed_diff:.6f}\n")
        f.write(f"perm_test_diff: {diff:.6f}, pval: {pval:.6f}\n")
        f.write(f"bootstrap_95_CI_diff: [{lower:.6f}, {upper:.6f}]\n")
        f.write(f"csv_file: {csv_path}\n")

    # print results
    print("=== SUMMARY ===")
    print(f"mean_baseline = {mean_baseline:.6f}")
    print(f"mean_intention = {mean_intention:.6f}")
    print(f"observed_diff = {observed_diff:.6f}")
    print(f"perm_test_diff = {diff:.6f}, p = {pval:.6f}")
    print(f"bootstrap_95_CI_diff = [{lower:.6f}, {upper:.6f}]")
    print(f"Data written to: {csv_path} and {summary_path}")

# --------------------------
# --- CLI ---
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=str, default="anon", help="participant id")
    parser.add_argument("--trials", type=int, default=TRIALS, help="number of trials")
    parser.add_argument("--visualize", action="store_true", help="show visualization (pygame)")
    parser.add_argument("--no-udp", action="store_true", help="disable UDP trigger")
    args = parser.parse_args()

    TRIALS = args.trials
    VISUALIZE = args.visualize or VISUALIZE
    if args.no_udp:
        UDP_ENABLE = False

    run_experiment(args)
