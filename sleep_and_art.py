import asyncio
import json
import time
import math
import random
import threading
import socket
from collections import deque
from datetime import datetime

import numpy as np
import pygame
import requests
from flask import Flask, jsonify, render_template_string
from bleak import BleakClient, BleakScanner

SERVICE_UUID = "4fa8c2d1-8253-4243-9780-7005a3674630"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

CYCLE_DURATION = 300.0  # 5 perc
CANVAS_W, CANVAS_H = 1000, 600

# ---------- AI / WEB beállítások ----------
FLASK_PORT       = 5000
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "gemma3:4b"
BUFFER_SIZE      = 1500   # mennyi nyers mintát tárolunk az elemzéshez
AI_COOLDOWN      = 35     # mp, ennyi időnként fut újra az AI
FEATURE_WINDOW_S = 30.0   # az elemzés az utolsó ennyi mp adatából számol

# Mozgás küszöbök (szögsebesség, °/s) – a szenzor rögzítéséhez igazítható
MOTION_LOW_DPS   = 3.0
MOTION_HIGH_DPS  = 15.0
TURN_SPEED_DPS   = 60.0   # ennél gyorsabb forgás = testhelyzet-váltás esemény

# 1. SZÍNVÁLASZTÁS
COLORS = {
    "1": ("Neon Ciano / Kék", (0, 255, 240)),
    "2": ("Neon Magenta / Pink", (255, 0, 128)),
    "3": ("Neon Zöld", (57, 255, 20)),
    "4": ("Neon Narancs", (255, 103, 0)),
    "5": ("Neon Sárga", (255, 240, 31)),
    "6": ("Elektromos Lila", (189, 0, 255)),
    "7": ("Fehér", (255, 255, 255)),
    "8": ("Fekete / Tus", (20, 20, 20)),
    "9": ("Klasszikus Piros", (230, 40, 40)),
    "10": ("Klasszikus Zöld", (40, 180, 60)),
    "11": ("Klasszikus Kék", (40, 90, 220)),
    "12": ("Rózsaszín", (240, 120, 180)),
    "13": ("Menta", (80, 220, 170)),
    "14": ("Arany", (212, 175, 55)),
    "15": ("Barna / Föld", (130, 75, 40)),
    "16": ("Szürke", (140, 140, 140)),
    "17": ("Korall", (255, 127, 80))
}

print("\n=== 1. SZÍNVÁLASZTÁS ===")
for key, (name, _) in COLORS.items():
    print(f"[{key}] {name}")
color_choice = input("\nVálassz egy színt (1-17) [Alapértelmezett: 1]: ").strip()
if color_choice not in COLORS:
    color_choice = "1"
SELECTED_COLOR_NAME, DRAW_COLOR = COLORS[color_choice]

# 2. MINTAVÁLASZTÁS
PATTERNS = {
    "1": ("SQUARE_SPIRAL", "Spirál Négyzetek (Szögletes organikus alakzatok)"),
    "2": ("NAUTILUS", "Nautilus (Kagylóhéj / Organikus spirál)"),
    "3": ("GALAXY_SPIRAL", "Galaxis Spirál (Örvénylő csillagköd)")
}

print("\n=== 2. MINTAVÁLASZTÁS ===")
for key, (_, desc) in PATTERNS.items():
    print(f"[{key}] {desc}")
pattern_choice = input("\nVálassz egy mintát (1-3) [Alapértelmezett: 1]: ").strip()
if pattern_choice not in PATTERNS:
    pattern_choice = "1"
SELECTED_PATTERN = PATTERNS[pattern_choice][0]

# 3. STÍLUSVÁLASZTÁS
STYLES = {
    "1": ("NEON_GLOW", "Cyberpunk Neon (Világító izzás sötét háttéren)"),
    "2": ("INK_CHARCOAL", "Klasszikus Tusrajz (Művészi papír és mély tónusok)"),
    "3": ("WATERCOLOR", "Akvarell (Lágy, áttetsző organikus rétegek)"),
    "4": ("MINIMAL_VECTOR", "Minimalista Vektor (Tűéles, sima vonalak)")
}

print("\n=== 3. STÍLUSVÁLASZTÁS ===")
for key, (name, desc) in STYLES.items():
    print(f"[{key}] {desc}")
style_choice = input("\nVálassz egy stílust (1-4) [Alapértelmezett: 1]: ").strip()
if style_choice not in STYLES:
    style_choice = "1"
SELECTED_STYLE = STYLES[style_choice][0]

# Háttérszín meghatározása stílus alapján
if SELECTED_STYLE == "NEON_GLOW":
    BG_COLOR = (10, 10, 18)
elif SELECTED_STYLE == "INK_CHARCOAL":
    BG_COLOR = (242, 238, 226)
elif SELECTED_STYLE == "WATERCOLOR":
    BG_COLOR = (248, 246, 240)
else:  # MINIMAL_VECTOR
    BG_COLOR = (255, 255, 255) if DRAW_COLOR != (255, 255, 255) else (20, 20, 20)

print(f"\n==========================================")
print(f" MINTA: {SELECTED_PATTERN} | STÍLUS: {SELECTED_STYLE} | SZÍN: {SELECTED_COLOR_NAME}")
print(f"==========================================\n")

NUM_PARTICLES = 80
pulse_min, pulse_max = 4095, 0
pitch_min, pitch_max = 90.0, -90.0

live_data = {
    "pulse_raw": 0,
    "pitch": 0.0,
    "bpm": 0,
    "pulse_norm": 0.5,
    "pitch_norm": 0.5
}

pulse_history = []
last_beat_time = time.time()

particle_positions = []
particle_velocities = []
paths = []

# =====================================================================
#  AI / WEB RÉSZ – a BLE-ről érkező adatokból (pulzus + kvaternió)
# =====================================================================

_lock         = threading.Lock()
_sleep_buffer = deque(maxlen=BUFFER_SIZE)
_latest       = {"pulse": 0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                 "pitch": 0.0, "roll": 0.0}


def push_sample(pulse_raw, qx, qy, qz, qw, pitch, roll):
    """A BLE notification handler hívja minden beérkező mintánál."""
    pkt = {"pulse": pulse_raw, "qx": qx, "qy": qy, "qz": qz, "qw": qw,
           "pitch": pitch, "roll": roll, "_ts": time.time()}
    with _lock:
        _latest.update(pkt)
        _sleep_buffer.append(pkt)


def get_latest():
    with _lock:
        return dict(_latest)


def get_sleep_buffer():
    with _lock:
        return list(_sleep_buffer)


def calculate_roll(qx, qy, qz, qw):
    return math.degrees(math.atan2(2.0 * (qw * qx + qy * qz),
                                   1.0 - 2.0 * (qx * qx + qy * qy)))


def _quat_angle_deg(a, b):
    """Két kvaternió közötti szögelfordulás fokban."""
    dot = (a["qx"] * b["qx"] + a["qy"] * b["qy"] +
           a["qz"] * b["qz"] + a["qw"] * b["qw"])
    dot = min(1.0, abs(dot))
    return math.degrees(2.0 * math.acos(dot))


_ai_lock        = threading.Lock()
_last_ai_time   = 0.0
_last_ai_result = {}
_ai_busy        = False


def _compute_sleep_features():
    data = get_sleep_buffer()
    if len(data) < 10:
        return None

    t_end = data[-1]["_ts"]
    win = [d for d in data if t_end - d["_ts"] <= FEATURE_WINDOW_S]
    n = len(win)
    if n < 10:
        return None

    pulses  = [d["pulse"] for d in win]
    pitches = [d["pitch"] for d in win]
    rolls   = [d["roll"]  for d in win]
    dur_s   = max(0.001, win[-1]["_ts"] - win[0]["_ts"])

    # --- pulzus statisztika (nyers ADC érték) ---
    pulse_mean  = sum(pulses) / n
    pulse_std   = math.sqrt(sum((p - pulse_mean) ** 2 for p in pulses) / n)
    pulse_range = max(pulses) - min(pulses)

    # --- mozgás: szögsebesség a kvaternió-változásokból ---
    speeds = []
    for i in range(1, n):
        dt_i = win[i]["_ts"] - win[i - 1]["_ts"]
        if dt_i > 0.001:
            speeds.append(_quat_angle_deg(win[i - 1], win[i]) / dt_i)
    motion_mean = sum(speeds) / len(speeds) if speeds else 0.0
    motion_max  = max(speeds) if speeds else 0.0

    # testhelyzet-váltás események (felfutó élek)
    turns = 0
    above = False
    for s in speeds:
        if s > TURN_SPEED_DPS and not above:
            turns += 1
            above = True
        elif s <= TURN_SPEED_DPS:
            above = False

    # --- szögek ---
    pitch_now = pitches[-1]
    roll_now  = rolls[-1]
    pitch_mean = sum(pitches) / n
    pitch_var  = sum((x - pitch_mean) ** 2 for x in pitches) / n

    # --- pulzus (BPM) csúcsdetektálással, időbélyegek alapján ---
    hr_bpm = None
    hrv_ms = None
    peak_times = []
    for i in range(20, n - 1):
        avg = sum(pulses[i - 20:i]) / 20.0
        if (pulses[i] > avg + 80 and pulses[i] >= pulses[i - 1]
                and pulses[i] > pulses[i + 1]):
            tt = win[i]["_ts"]
            if not peak_times or (tt - peak_times[-1]) > 0.4:
                peak_times.append(tt)
    if len(peak_times) >= 3:
        ivs = [peak_times[j + 1] - peak_times[j] for j in range(len(peak_times) - 1)]
        mean_iv = sum(ivs) / len(ivs)
        cand = round(60.0 / mean_iv)
        if 30 < cand < 200:
            hr_bpm = cand
            hrv_ms = round(math.sqrt(sum((x - mean_iv) ** 2 for x in ivs) / len(ivs)) * 1000, 1)

    motion = ("high" if motion_mean > MOTION_HIGH_DPS
              else "low" if motion_mean < MOTION_LOW_DPS else "moderate")

    return {
        "motion_dps":  round(motion_mean, 2),
        "motion_max":  round(motion_max, 1),
        "pitch_var":   round(pitch_var, 3),
        "pitch":       round(pitch_now, 1),
        "roll":        round(roll_now, 1),
        "turns":       turns,
        "pulse_mean":  round(pulse_mean, 1),
        "pulse_std":   round(pulse_std, 1),
        "pulse_range": int(pulse_range),
        "hr_bpm":      hr_bpm,
        "hrv_ms":      hrv_ms,
        "samples":     n,
        "window_s":    round(dur_s, 1),
        "motion":      motion,
    }


def _ollama_analyze(features):
    global _ai_busy
    _ai_busy = True
    hr_s  = f"{features['hr_bpm']} bpm" if features['hr_bpm'] else "not available"
    hrv_s = f"{features['hrv_ms']} ms" if features['hrv_ms'] is not None else "not available"

    prompt = f"""You are a sleep analysis AI. Analyze the sleep state based on the following sensor data.

SENSOR DATA (last ~{features['window_s']} sec):
- Motion intensity (mean angular speed): {features['motion_dps']} °/s  [{features['motion']}]
- Peak angular speed: {features['motion_max']} °/s
- Pitch variance: {features['pitch_var']}
- Pitch angle: {features['pitch']}°
- Roll angle: {features['roll']}°
- Position changes: {features['turns']}
- Pulse (estimated heart rate): {hr_s}
- Beat interval variability (SDNN): {hrv_s}
- Raw pulse signal mean / std / range: {features['pulse_mean']} / {features['pulse_std']} / {features['pulse_range']}
- Sample count: {features['samples']}

GUIDELINES:
- Motion < {MOTION_LOW_DPS} °/s and low pitch variance = deep sleep or REM
- Motion {MOTION_LOW_DPS}-{MOTION_HIGH_DPS} °/s = light sleep
- Motion > {MOTION_HIGH_DPS} °/s = awake or restless
- High beat interval variability with very low motion can indicate REM
- Pitch ~0° and roll ~0° = lying on back
- Roll > 45° = right side, Roll < -45° = left side
- Pitch > 30° = lying on stomach

Reply ONLY with valid JSON, no other text, explanation or formatting:
{{"sleep_stage":"awake|light sleep|deep sleep|REM","quality_score":0-100,"body_position":"on back|left side|right side|on stomach","restlessness":"calm|moderate|restless","summary":"1-2 sentence English summary of the current state","tips":["tip1","tip2"]}}"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.15, "num_predict": 350, "top_p": 0.9}
        }, timeout=90)
        raw = resp.json().get("response", "").strip()

        result = {"error": "JSON not found", "raw": raw[:200]}
        start = raw.find("{")
        if start >= 0:
            depth = 0
            end = -1
            for i, ch in enumerate(raw[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    result = json.loads(raw[start:end])
                except json.JSONDecodeError as je:
                    result = {"error": f"JSON parse error: {je}", "raw": raw[start:end][:200]}
    except requests.exceptions.Timeout:
        result = {"error": "Ollama időtúllépés (90s) – modell még tölt?"}
    except Exception as ex:
        result = {"error": str(ex)}

    _ai_busy = False
    return result


def get_sleep_analysis():
    global _last_ai_time, _last_ai_result
    features = _compute_sleep_features()
    if features is None:
        return None, None, True
    now    = time.time()
    cached = (now - _last_ai_time) < AI_COOLDOWN or _ai_busy
    if not cached:
        _last_ai_result = _ollama_analyze(features)
        _last_ai_time   = time.time()
    return features, _last_ai_result, cached


flask_app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SleepArt – Sleep Analyzer</title>

    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">

    <style>
        :root {
            --bg: #080c14;
            --surface: #0e1623;
            --border: #1a2540;
            --muted: #3a4d6e;
            --text: #c8d8f0;
            --dim: #6b82a8;
            --blue: #4d9fff;
            --violet: #9b7fff;
            --teal: #2dd4c0;
            --red: #ff5f6d;
            --green: #36d86e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
            min-height: 100vh;
        }

        header {
            display: flex;
            align-items: center;
            gap: .8rem;
            padding: 1.25rem 2rem;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            background: var(--bg);
            z-index: 10;
        }

        .logo {
            font-family: 'Space Mono', monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--blue);
        }

        .live-pill {
            display: flex;
            align-items: center;
            gap: .45rem;
            background: #0a1f10;
            border: 1px solid #1a4028;
            border-radius: 20px;
            padding: .25rem .75rem;
            font-size: .72rem;
            font-family: 'Space Mono', monospace;
            color: var(--green);
        }

        .live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
            animation: blink 2s ease-in-out infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: .25; }
        }

        .upd-time {
            margin-left: auto;
            font-size: .72rem;
            font-family: 'Space Mono', monospace;
            color: var(--muted);
        }

        .stage-hero {
            padding: 2rem;
            display: flex;
            gap: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        .stage-icon {
            font-size: 3rem;
        }

        .stage-eyebrow {
            font-size: .68rem;
            font-family: 'Space Mono';
            color: var(--dim);
            text-transform: uppercase;
        }

        .stage-name {
            font-size: 2.4rem;
            font-weight: 700;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: .75rem;
            padding: 1.5rem 2rem;
        }

        .metric {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
        }

        .metric-label {
            font-size: .65rem;
            font-family: 'Space Mono';
            color: var(--dim);
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            font-family: 'Space Mono';
            color: var(--blue);
        }

        .ai-section {
            margin: 0 2rem 1.5rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        .ai-header {
            display: flex;
            padding: .9rem 1.25rem;
            border-bottom: 1px solid var(--border);
            font-family: 'Space Mono';
            font-size: .7rem;
        }

        .refresh-btn {
            margin: 0 2rem 2rem;
            width: calc(100% - 4rem);
            padding: .75rem;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--dim);
            border-radius: 8px;
        }
    </style>
</head>

<body>

<header>
    <div class="logo">SleepArt · Sleep Analyzer</div>

    <div class="live-pill">
        <div class="live-dot"></div>
        <span id="buf-count">0 samples</span>
    </div>

    <span class="upd-time" id="upd-time">–</span>
</header>

<div class="stage-hero" id="hero" style="display:none">
    <div class="stage-icon" id="stage-icon">😴</div>

    <div>
        <div class="stage-eyebrow">Sleep stage</div>
        <div class="stage-name" id="stage-name">–</div>
    </div>
</div>

<div class="metrics" id="metrics" style="display:none">
    <div class="metric">
        <div class="metric-label">Motion (SMA)</div>
        <div class="metric-value" id="m-sma">–</div>
    </div>
</div>

<div class="ai-section" id="ai-section" style="display:none">
    <div class="ai-header">
        AI analysis · gemma3:4b · ollama
    </div>
</div>

<button class="refresh-btn" id="ref-btn" onclick="load()" style="display:none">
    Refresh analysis now
</button>

<script>
    const STAGE_ICONS = {
        "awake": "👀",
        "light sleep": "🌙",
        "deep sleep": "💤",
        "REM": "🌀"
    };

    function set(id, v) {
        document.getElementById(id).textContent = v;
    }

    async function load() {
        const r = await fetch("/api/analysis");
        const d = await r.json();

        set("buf-count", d.buf_size + " samples");
        set("upd-time", new Date().toLocaleTimeString("en-GB"));
    }

    load();
    setInterval(load, 30000);
</script>

</body>
</html>
"""


@flask_app.route("/")
def flask_index():
    return render_template_string(DASHBOARD_HTML)


@flask_app.route("/api/analysis")
def flask_analysis():
    global _last_ai_time
    features, ai, cached = get_sleep_analysis()
    if features is None:
        return jsonify({"error": "Not enough data – wait a few seconds."})
    return jsonify({
        "features":  features,
        "ai":        ai,
        "cached":    cached,
        "ai_age_s":  round(time.time() - _last_ai_time),
        "buf_size":  len(_sleep_buffer),
        "timestamp": datetime.now().isoformat(),
    })


@flask_app.route("/api/raw")
def flask_raw():
    return jsonify(get_sleep_buffer()[-60:])


@flask_app.route("/api/status")
def flask_status():
    return jsonify({
        "buffer_len":    len(_sleep_buffer),
        "ollama_model":  OLLAMA_MODEL,
        "ai_busy":       _ai_busy,
        "last_ai_age_s": round(time.time() - _last_ai_time),
        "latest_packet": get_latest(),
    })


def _run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=FLASK_PORT,
                  debug=False, use_reloader=False)


def start_web():
    threading.Thread(target=_run_flask, daemon=True).start()
    time.sleep(0.3)  # Flask indulási idő
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _my_ip = _s.getsockname()[0]
        _s.close()
        print(f"[Flask] *** Sleep Analyzer: http://{_my_ip}:{FLASK_PORT} ***")
        print(f"[Flask] Ha nem éred el: sudo ufw allow {FLASK_PORT}")
    except Exception as _e:
        print(f"[Flask] IP detektálás sikertelen: {_e}")
        print(f"[Flask] Próbáld: hostname -I")


# =====================================================================
#  ALKOTÁS RÉSZ
# =====================================================================

def init_particles_for_pattern():
    global particle_positions, particle_velocities, paths
    particle_positions.clear()
    particle_velocities.clear()
    paths.clear()

    for i in range(NUM_PARTICLES):
        if SELECTED_PATTERN == "SQUARE_SPIRAL":
            # Négyzet alakú elrendezésből induló szálak
            side = i % 4
            layer = (i // 4) * 12 + 15
            if side == 0:
                px, py = CANVAS_W / 2 - layer, CANVAS_H / 2 - layer
                vx, vy = 1.5, 0.0
            elif side == 1:
                px, py = CANVAS_W / 2 + layer, CANVAS_H / 2 - layer
                vx, vy = 0.0, 1.5
            elif side == 2:
                px, py = CANVAS_W / 2 + layer, CANVAS_H / 2 + layer
                vx, vy = -1.5, 0.0
            else:
                px, py = CANVAS_W / 2 - layer, CANVAS_H / 2 + layer
                vx, vy = 0.0, -1.5

        elif SELECTED_PATTERN == "NAUTILUS":
            angle = (i / NUM_PARTICLES) * 4 * math.pi
            r = 10 + i * 2.5
            px = CANVAS_W / 2 + math.cos(angle) * r
            py = CANVAS_H / 2 + math.sin(angle) * r
            vx, vy = math.sin(angle) * 1.2, -math.cos(angle) * 1.2

        else:  # GALAXY_SPIRAL
            arm = i % 4
            angle = (i / NUM_PARTICLES) * 4 * math.pi
            r = 8 + i * 3
            px = CANVAS_W / 2 + math.cos(angle + (arm * math.pi / 2)) * r
            py = CANVAS_H / 2 + math.sin(angle + (arm * math.pi / 2)) * r
            vx, vy = math.sin(angle) * 1.5, -math.cos(angle) * 1.5

        pos = np.array([px, py], dtype=float)
        vel = np.array([vx, vy], dtype=float)
        particle_positions.append(pos)
        particle_velocities.append(vel)
        paths.append([pos.copy()])

init_particles_for_pattern()
start_time = None

def calculate_pitch(qx, qy, qz, qw):
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        return math.copysign(90.0, sinp)
    return math.degrees(math.asin(sinp))

def process_sensor_data(pulse_raw, pitch_deg, elapsed):
    global pulse_min, pulse_max, pitch_min, pitch_max, last_beat_time

    if pulse_raw < pulse_min: pulse_min = pulse_raw
    if pulse_raw > pulse_max: pulse_max = pulse_raw
    if pitch_deg < pitch_min: pitch_min = pitch_deg
    if pitch_deg > pitch_max: pitch_max = pitch_deg

    p_denom = max(1, (pulse_max - pulse_min))
    pulse_norm = (pulse_raw - pulse_min) / p_denom

    pitch_denom = max(1.0, (pitch_max - pitch_min))
    pitch_norm = (pitch_deg - pitch_min) / pitch_denom

    live_data["pulse_raw"] = pulse_raw
    live_data["pitch"] = round(pitch_deg, 1)
    live_data["pulse_norm"] = pulse_norm
    live_data["pitch_norm"] = pitch_norm

    pulse_history.append(pulse_raw)
    if len(pulse_history) > 20: pulse_history.pop(0)
    avg_p = sum(pulse_history) / len(pulse_history)
    if pulse_raw > avg_p + 80 and (time.time() - last_beat_time) > 0.4:
        live_data["bpm"] = int(60.0 / (time.time() - last_beat_time))
        last_beat_time = time.time()

def update_physics(elapsed):
    dt = 0.4
    p_norm = live_data["pulse_norm"]
    pitch_deg = live_data["pitch"]
    t = time.time()

    for i in range(NUM_PARTICLES):
        pos = particle_positions[i]
        vel = particle_velocities[i]

        if SELECTED_PATTERN == "SQUARE_SPIRAL":
            cx, cy = CANVAS_W / 2 + pitch_deg * 2.0, CANVAS_H / 2
            dx, dy = pos[0] - cx, pos[1] - cy

            # Négyzetes kanyarodási logika
            if abs(dx) > abs(dy):
                vx = -np.sign(dy) * (1.2 + p_norm * 1.5)
                vy = np.sign(dx) * (1.2 + p_norm * 1.5)
            else:
                vx = np.sign(dy) * (1.2 + p_norm * 1.5)
                vy = -np.sign(dx) * (1.2 + p_norm * 1.5)

            # Finom tágulás/összehúzódás pulzus hatására
            vx += (dx / (abs(dx) + 1)) * (p_norm - 0.5)
            vy += (dy / (abs(dy) + 1)) * (p_norm - 0.5)
            vel = np.array([vx, vy])

        elif SELECTED_PATTERN == "NAUTILUS":
            cx, cy = CANVAS_W / 2 + pitch_deg * 2.5, CANVAS_H / 2
            rx, ry = pos[0] - cx, pos[1] - cy
            dist = math.sqrt(rx**2 + ry**2) + 0.1
            rot = 0.04 + p_norm * 0.08
            vx = -ry / dist * (2.0 + rot * 8) + (rx / dist) * (0.3 + pitch_deg * 0.01)
            vy = rx / dist * (2.0 + rot * 8) + (ry / dist) * (0.3 + pitch_deg * 0.01)
            vel = np.array([vx, vy])

        else:  # GALAXY_SPIRAL
            cx, cy = CANVAS_W / 2 + pitch_deg * 2.0, CANVAS_H / 2
            rx, ry = pos[0] - cx, pos[1] - cy
            dist = math.sqrt(rx**2 + ry**2) + 0.1
            expansion = math.sin(t * 3.0) * (p_norm * 1.5)
            vel[0] = -ry / dist * (2.0 + p_norm * 1.5) + (rx / dist) * expansion
            vel[1] = rx / dist * (2.0 + p_norm * 1.5) + (ry / dist) * expansion

        pos += vel * dt
        pos[0] = max(10, min(CANVAS_W - 10, pos[0]))
        pos[1] = max(10, min(CANVAS_H - 10, pos[1]))

        particle_positions[i] = pos
        particle_velocities[i] = vel
        paths[i].append(pos.copy())

def render_artwork(surface):
    """Kirendereli az alkotást a kiválasztott művészeti stílus szerint."""
    surface.fill(BG_COLOR)

    if SELECTED_STYLE == "NEON_GLOW":
        # Multi-pass rendering a neon izzó hatás eléréséhez
        glow_surface = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)

        # 1. Külső széles izzás
        for path in paths:
            if len(path) > 1:
                pts = [(int(p[0]), int(p[1])) for p in path]
                r, g, b = DRAW_COLOR
                pygame.draw.lines(glow_surface, (r, g, b, 30), False, pts, 6)
                pygame.draw.lines(glow_surface, (r, g, b, 70), False, pts, 3)
                pygame.draw.lines(glow_surface, (255, 255, 255, 200), False, pts, 1)
        surface.blit(glow_surface, (0, 0))

    elif SELECTED_STYLE == "WATERCOLOR":
        # Áttetsző, lágy akvarell rétegek
        water_surface = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
        r, g, b = DRAW_COLOR
        for path in paths:
            if len(path) > 1:
                pts = [(int(p[0]), int(p[1])) for p in path]
                pygame.draw.lines(water_surface, (r, g, b, 40), False, pts, 3)
                pygame.draw.lines(water_surface, (r, g, b, 90), False, pts, 1)
        surface.blit(water_surface, (0, 0))

    elif SELECTED_STYLE == "INK_CHARCOAL":
        # Sötét tus hatás enyhén lágyított vonalakkal
        ink_surface = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
        r, g, b = DRAW_COLOR
        for path in paths:
            if len(path) > 1:
                pts = [(int(p[0]), int(p[1])) for p in path]
                pygame.draw.lines(ink_surface, (r, g, b, 180), False, pts, 2)
                pygame.draw.lines(ink_surface, (r, g, b, 240), False, pts, 1)
        surface.blit(ink_surface, (0, 0))

    else:  # MINIMAL_VECTOR
        # Tűéles, sima 1 pixeles vonalak
        for path in paths:
            if len(path) > 1:
                pts = [(int(p[0]), int(p[1])) for p in path]
                pygame.draw.lines(surface, DRAW_COLOR, False, pts, 1)

def save_png():
    """Elmenti a kész képet PNG formátumban a kívánt fájlnéven."""
    filename = input("\nAdandó fájlnév (kiterjesztés nélkül) [Alapértelmezett: művészi_alkotás]: ").strip()
    if not filename:
        filename = "művészi_alkotás"

    if not filename.endswith(".png"):
        filename += ".png"

    export_surface = pygame.Surface((CANVAS_W, CANVAS_H))
    render_artwork(export_surface)

    try:
        pygame.image.save(export_surface, filename)
        print(f"\n[SIKER] A kép sikeresen elmentve PNG-be: {filename}")
    except Exception as e:
        print(f"\n[HIBA] Nem sikerült menteni a PNG képet: {e}")

def notification_handler(sender, data: bytearray):
    global start_time
    if start_time is None:
        start_time = time.time()

    elapsed = time.time() - start_time
    if elapsed > CYCLE_DURATION: return

    try:
        payload = json.loads(data.decode('utf-8'))
        pulse_raw = payload.get("pulse", 0)
        qx, qy, qz, qw = payload.get("qx", 0), payload.get("qy", 0), payload.get("qz", 0), payload.get("qw", 1)

        pitch = calculate_pitch(qx, qy, qz, qw)
        roll = calculate_roll(qx, qy, qz, qw)

        # AI / web puffer feltöltése ugyanezekkel az adatokkal
        push_sample(pulse_raw, qx, qy, qz, qw, pitch, roll)

        process_sensor_data(pulse_raw, pitch, elapsed)
        update_physics(elapsed)
    except Exception:
        pass

async def main():
    global start_time
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption(f"Generatív Művészet - {SELECTED_PATTERN} ({SELECTED_STYLE})")
    font = pygame.font.SysFont("monospace", 14, bold=True)

    print("ESP32-S3 keresése BLE-n...")
    device = await BleakScanner.find_device_by_name("ESP32S3_Sensors", timeout=10.0)
    if not device:
        print("Hiba: Nem található az 'ESP32S3_Sensors' BLE eszköz!")
        return

    try:
        async with BleakClient(device) as client:
            print("Sikeres csatlakozás! Élő rajzolás indítása...")
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False

                render_artwork(screen)

                elapsed = int(time.time() - start_time) if start_time else 0

                txt_info = font.render(f"MINTA: {SELECTED_PATTERN} | STÍLUS: {SELECTED_STYLE}", True, (180, 180, 180))
                txt_time = font.render(f"IDŐ: {elapsed}s/300s", True, (220, 80, 80))

                s = pygame.Surface((CANVAS_W, 30), pygame.SRCALPHA)
                s.fill((0, 0, 0, 160))
                screen.blit(s, (0, 0))

                screen.blit(txt_info, (10, 6))
                screen.blit(txt_time, (820, 6))

                pygame.display.flip()

                try:
                    await asyncio.sleep(0.03)
                except asyncio.CancelledError:
                    break

                if start_time and (time.time() - start_time) >= CYCLE_DURATION:
                    running = False

    except Exception:
        pass
    finally:
        pygame.quit()
        save_png()

if __name__ == "__main__":
    start_web()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError, Exception):
        pass
