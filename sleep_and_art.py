import socket, json, math, random, os, time, threading
from collections import deque
from datetime import datetime

import pygame
import requests
from flask import Flask, jsonify, render_template_string

UDP_PORT         = 5005
FLASK_PORT       = 5000
SESSION_SECONDS  = 300          
CYCLE_SECONDS    = 10           
SAMPLE_RATE      = 20
BPM_WINDOW       = 5
SAVE_DIR         = "cosmos_output"
os.makedirs(SAVE_DIR, exist_ok=True)

TURN_GYRO_MULT     = 3.0
TURN_GYRO_MIN      = 18.0
METEOR_COOLDOWN    = 3.0
MAX_METEORS        = 1

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
BUFFER_SIZE  = 600   
AI_COOLDOWN  = 35

_lock         = threading.Lock()
_ir_buffer    = deque(maxlen=SAMPLE_RATE * BPM_WINDOW)
_red_buffer   = deque(maxlen=SAMPLE_RATE * BPM_WINDOW)
_sleep_buffer = deque(maxlen=BUFFER_SIZE)
_latest       = {"ax": 0.0, "ay": 0.0, "az": 1.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "red": 0,  "ir": 0}

def _udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"[UDP] Listening on port {UDP_PORT}")
    while True:
        try:
            raw, _ = sock.recvfrom(512)
            pkt = json.loads(raw.decode())
            pkt["_ts"] = time.time()
            with _lock:
                _latest.update(pkt)
                _sleep_buffer.append(dict(pkt))
                if pkt.get("ir", 0) > 5000:
                    _ir_buffer.append(pkt["ir"])
                    _red_buffer.append(pkt.get("red", 0))
        except Exception:
            pass

threading.Thread(target=_udp_listener, daemon=True).start()

def get_latest():
    with _lock:
        return dict(_latest)

def get_sleep_buffer():
    with _lock:
        return list(_sleep_buffer)

def compute_bpm():
    with _lock:
        samples = list(_ir_buffer)
    if len(samples) < 40:
        return 72
    mean = sum(samples) / len(samples)
    ac   = [s - mean for s in samples]
    thr  = max(ac) * 0.30
    peaks = [i for i in range(1, len(ac) - 1)
             if ac[i] > thr and ac[i] > ac[i-1] and ac[i] > ac[i+1]]
    if len(peaks) < 2:
        return 72
    avg_iv = (sum(peaks[j+1] - peaks[j] for j in range(len(peaks)-1))
              / (len(peaks)-1)) / SAMPLE_RATE
    return max(40, min(200, int(60.0 / avg_iv)))

def compute_spo2():
    with _lock:
        ir_l, red_l = list(_ir_buffer), list(_red_buffer)
    if len(ir_l) < 40 or len(red_l) < 40:
        return 97
    dc_ir, dc_red = sum(ir_l) / len(ir_l), sum(red_l) / len(red_l)
    if dc_ir < 1 or dc_red < 1:
        return 97
    ac_ir, ac_red = max(ir_l) - min(ir_l), max(red_l) - min(red_l)
    r = (ac_red / dc_red) / (ac_ir / dc_ir)
    return max(85, min(100, int(110 - 25 * r)))

def gyro_magnitude(p):
    return math.sqrt(p.get("gx", 0)**2 + p.get("gy", 0)**2 + p.get("gz", 0)**2)

def accel_horizontal(p):
    return math.sqrt(p.get("ax", 0)**2 + p.get("ay", 0)**2)

_ai_lock        = threading.Lock()
_last_ai_time   = 0.0
_last_ai_result = {}
_ai_busy        = False

def _compute_sleep_features():
    data = get_sleep_buffer()
    n    = len(data)
    if n < 10:
        return None

    ax_l = [d["ax"] for d in data]
    ay_l = [d["ay"] for d in data]
    az_l = [d["az"] for d in data]
    gx_l = [d.get("gx", 0) for d in data]
    gy_l = [d.get("gy", 0) for d in data]
    gz_l = [d.get("gz", 0) for d in data]

    sma      = sum(abs(ax_l[i]) + abs(ay_l[i]) + abs(az_l[i]) for i in range(n)) / n
    ax_mean  = sum(ax_l) / n
    variance = sum((x - ax_mean) ** 2 for x in ax_l) / n
    gyro_mag = sum(math.sqrt(gx_l[i]**2 + gy_l[i]**2 + gz_l[i]**2)
                   for i in range(n)) / n

    try:
        pitch = math.atan2(ax_l[-1],
                math.sqrt(ay_l[-1]**2 + az_l[-1]**2)) * 180 / math.pi
        roll  = math.atan2(ay_l[-1], az_l[-1]) * 180 / math.pi
    except Exception:
        pitch = roll = 0.0

    turns = sum(1 for i in range(1, n)
                if abs(ax_l[i] - ax_l[i-1]) > 0.25
                or abs(ay_l[i] - ay_l[i-1]) > 0.25)

    ir_l  = [d["ir"]        for d in data if d.get("ir",  0) > 1000]
    red_l = [d.get("red",0) for d in data if d.get("ir",  0) > 1000]
    spo2_val = hr_bpm = None

    if len(ir_l) >= 10:
        ir_mean  = sum(ir_l)  / len(ir_l)
        red_mean = sum(red_l) / len(red_l)
        if ir_mean > 0:
            spo2_val = round(max(85.0, min(100.0, 110.0 - 25.0 * (red_mean / ir_mean))), 1)
        if len(ir_l) >= 20:
            thr   = ir_mean * 1.005
            peaks = sum(1 for i in range(1, len(ir_l)-1)
                        if ir_l[i] > thr
                        and ir_l[i] > ir_l[i-1]
                        and ir_l[i] > ir_l[i+1])
            dur_s = len(ir_l) / 20.0
            if dur_s > 2:
                cand   = round(peaks / dur_s * 60)
                hr_bpm = cand if 30 < cand < 200 else None

    motion = ("high" if sma > 0.10 else "low" if sma < 0.02 else "moderate")

    return {
        "sma":         round(sma, 4),
        "variance":    round(variance, 5),
        "gyro_mag":    round(gyro_mag, 2),
        "pitch":       round(pitch, 1),
        "roll":        round(roll, 1),
        "turns":       turns,
        "spo2":        spo2_val,
        "hr_bpm":      hr_bpm,
        "samples":     n,
        "motion":      motion,
        "has_optical": len(ir_l) > 0,
    }

def _ollama_analyze(features):
    global _ai_busy
    _ai_busy = True
    spo2_s = f"{features['spo2']} %" if features['spo2'] else "not available"
    hr_s   = f"{features['hr_bpm']} bpm" if features['hr_bpm'] else "not available"

    prompt = f"""You are a sleep analysis AI. Analyze the sleep state based on the following sensor data.

SENSOR DATA (last ~30 sec):
- Motion intensity (SMA): {features['sma']} g  [{features['motion']}]
- Motion variance: {features['variance']}
- Gyroscope activity: {features['gyro_mag']} °/s
- Pitch angle: {features['pitch']}°
- Roll angle: {features['roll']}°
- Position changes: {features['turns']} db
- SpO2: {spo2_s}
- Pulzus: {hr_s}
- Sample count: {features['samples']}

GUIDELINES:
- SMA < 0.02 and variance < 0.001 = deep sleep or REM
- SMA 0.02–0.08 = light sleep
- SMA > 0.08 = awake or restless
- Pitch ~0° and roll ~0° = lying on back
- Roll > 45° = right side, Roll < -45° = left side
- Pitch > 30° = lying on stomach

Reply ONLY with valid JSON, no other text, explanation or formatting:
{{"sleep_stage":"awake|light sleep|deep sleep|REM","quality_score":0-100,"body_position":"on back|left side|right side|on stomach","restlessness":"calm|moderate|restless","spo2_status":"normal|low|not measurable","summary":"1-2 sentence English summary of the current state","tips":["tip1","tip2"]}}"""

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

threading.Thread(target=_run_flask, daemon=True).start()
time.sleep(0.3)  # Flask indulási idő
import socket as _sock
try:
    _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
    _s.connect(("8.8.8.8", 80))
    _my_ip = _s.getsockname()[0]
    _s.close()
    print(f"[Flask] *** Sleep Analyzer: http://{_my_ip}:{FLASK_PORT} ***")
    print(f"[Flask] Ha nem éred el: sudo ufw allow {FLASK_PORT}")
except Exception as _e:
    print(f"[Flask] IP detektálás sikertelen: {_e}")
    print(f"[Flask] Próbáld: hostname -I")

pygame.init()
pygame.mouse.set_visible(False)
if os.environ.get("SLEEPART_WINDOWED") == "1":
    screen = pygame.display.set_mode((1280, 720))
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
CENTER = (WIDTH // 2, HEIGHT // 2)
pygame.display.set_caption("SleepArt · Cosmos")
clock = pygame.time.Clock()

FONT_BIG   = pygame.font.SysFont("monospace", 46, bold=True)
FONT_MED   = pygame.font.SysFont("monospace", 24, bold=True)
FONT_SMALL = pygame.font.SysFont("monospace", 16)

def lerp(a, b, t): return a + (b - a) * t

def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))

def brighten(c, f=1.6):
    return tuple(min(255, int(v * f)) for v in c)

PALETTES = {
    "awake": [(255, 150, 95), (255, 205, 120), (255, 95, 150)],
    "light": [(120, 180, 255), (165, 140, 255), (80, 225, 220)],
    "deep":  [(70, 60, 165), (35, 35, 115), (115, 70, 205)],
    "rem":   [(255, 95, 220), (120, 255, 210), (165, 120, 255)],
    "alert": [(255, 70, 70), (255, 140, 60), (235, 30, 90)],
}

def palette_color(stage, t):
    colors = PALETTES[stage]
    n = len(colors)
    pos = (t * 0.09) % n
    i = int(pos)
    return lerp_color(colors[i], colors[(i + 1) % n], pos - i)

def make_gradient(w, h, top, bottom):
    surf = pygame.Surface((w, h))
    for y in range(h):
        pygame.draw.line(surf, lerp_color(top, bottom, y / h), (0, y), (w, y))
    return surf

BG_GRADIENT = make_gradient(WIDTH, HEIGHT, (10, 7, 24), (3, 3, 11))
nebula_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
star_layer   = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
fx_layer     = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

class Star:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.r = random.uniform(0.6, 2.1)
        self.phase = random.uniform(0, math.tau)
        self.speed = random.uniform(0.3, 0.9)
        self.drift = random.uniform(1.5, 6.0)

    def draw(self, t):
        a = int(70 + 150 * (0.5 + 0.5 * math.sin(t * self.speed + self.phase)))
        pygame.draw.circle(star_layer, (210, 220, 255, a),
                            (int(self.x), int(self.y)), max(1, int(self.r)))
        self.x += self.drift * 0.02
        if self.x > WIDTH + 4:
            self.x = -4

STARS = [Star() for _ in range(160)]

NEBULA_FADE = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
NEBULA_FADE.fill((250, 250, 250, 250))

def glow(target, pos, radius, color, layers=6, max_alpha=150):
    radius = max(2, int(radius))
    x, y = int(pos[0]), int(pos[1])
    for i in range(layers, 0, -1):
        r = int(radius * i / layers)
        if r <= 0:
            continue
        a = int(max_alpha * (1 - i / layers) ** 1.3) + 6
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, a), (r, r), r)
        target.blit(s, (x - r, y - r), special_flags=pygame.BLEND_RGBA_ADD)

def add_nebula_bloom(pos, color, radius):
    glow(nebula_layer, pos, radius, color, layers=7, max_alpha=80)

class Ring:
    def __init__(self, center, color, max_r, speed=130, width=3):
        self.cx, self.cy = center
        self.r = 4.0
        self.color = color
        self.max_r = max_r
        self.speed = speed
        self.width = width

    def update(self, dt):
        self.r += self.speed * dt

    @property
    def alive(self):
        return self.r < self.max_r

    def draw(self, surf):
        t = self.r / self.max_r
        a = max(0, int(210 * (1 - t)))
        if a > 1:
            pygame.draw.circle(surf, (*self.color, a),
                                (int(self.cx), int(self.cy)), int(self.r), self.width)

class Spark:
    def __init__(self, pos, color, speed_range=(60, 260)):
        ang = random.uniform(0, math.tau)
        spd = random.uniform(*speed_range)
        self.x, self.y = pos
        self.vx, self.vy = math.cos(ang) * spd, math.sin(ang) * spd
        self.color = color
        self.life = random.uniform(0.5, 1.3)
        self.age = 0.0
        self.r = random.uniform(1.5, 3.5)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.96
        self.vy *= 0.96

    @property
    def alive(self):
        return self.age < self.life

    def draw(self, surf):
        t = self.age / self.life
        a = max(0, int(255 * (1 - t)))
        if a > 1:
            pygame.draw.circle(surf, (*self.color, a), (int(self.x), int(self.y)),
                                max(1, int(self.r * (1 - t * 0.6))))

class Meteor:
    def __init__(self, start, angle, distance, color, speed, size):
        self.x, self.y = start
        self.vx, self.vy = math.cos(angle) * speed, math.sin(angle) * speed
        self.tx = start[0] + math.cos(angle) * distance
        self.ty = start[1] + math.sin(angle) * distance
        self.color = color
        self.size = size
        self.trail = deque(maxlen=10)
        self.exploded = False

    def update(self, dt):
        self.trail.append((self.x, self.y))
        self.x += self.vx * dt
        self.y += self.vy * dt
        if math.hypot(self.x - self.tx, self.y - self.ty) < self.size * 1.4:
            self.exploded = True

    def draw(self, surf):
        n = len(self.trail)
        for i, (px, py) in enumerate(self.trail):
            a = int(170 * (i / max(1, n)))
            r = max(1, int(self.size * 0.5 * (i / max(1, n))))
            pygame.draw.circle(surf, (*self.color, a), (int(px), int(py)), r)
        glow(surf, (self.x, self.y), self.size * 1.6, brighten(self.color, 1.2), layers=4)

rings, sparks, meteors = [], [], []
last_meteor_t  = 0.0

def classify_stage(elapsed, motion_avg, bpm_recent):
    if elapsed < 25:
        return "awake"
    if motion_avg > 0.07:
        return "awake"
    if motion_avg > 0.025:
        return "light"
    if len(bpm_recent) >= 3:
        m = sum(bpm_recent) / len(bpm_recent)
        var = sum((b - m) ** 2 for b in bpm_recent) / len(bpm_recent)
        if var > 18:
            return "rem"
    return "deep"

class CycleAverager:
    def __init__(self):
        self.samples = {"motion": [], "bpm": [], "spo2": []}
        self.cycle_start = time.time()
        self.prev = {"motion": 0.02, "bpm": 70, "spo2": 97, "stage": "awake"}
        self.cur  = dict(self.prev)
        self.bpm_recent = deque(maxlen=12)
        self.log = []

    def add_sample(self, motion, bpm, spo2):
        self.samples["motion"].append(motion)
        self.samples["bpm"].append(bpm)
        self.samples["spo2"].append(spo2)

    def maybe_roll(self, elapsed):
        if time.time() - self.cycle_start < CYCLE_SECONDS:
            return
        self.prev = dict(self.cur)
        avg_motion = sum(self.samples["motion"]) / max(1, len(self.samples["motion"]))
        avg_bpm    = sum(self.samples["bpm"])    / max(1, len(self.samples["bpm"]))
        avg_spo2   = sum(self.samples["spo2"])   / max(1, len(self.samples["spo2"]))
        self.bpm_recent.append(avg_bpm)
        stage = classify_stage(elapsed, avg_motion, list(self.bpm_recent))
        self.cur = {"motion": avg_motion, "bpm": avg_bpm, "spo2": avg_spo2, "stage": stage}
        self.log.append({"t": round(elapsed), **{k: round(v, 2) if isinstance(v, float) else v
                                                   for k, v in self.cur.items()}})
        self.samples = {"motion": [], "bpm": [], "spo2": []}
        self.cycle_start = time.time()

    def smoothed(self):
        progress = min(1.0, (time.time() - self.cycle_start) / CYCLE_SECONDS)
        motion = lerp(self.prev["motion"], self.cur["motion"], progress)
        bpm    = lerp(self.prev["bpm"],    self.cur["bpm"],    progress)
        spo2   = lerp(self.prev["spo2"],   self.cur["spo2"],   progress)
        stage  = self.cur["stage"] if progress > 0.5 else self.prev["stage"]
        return motion, bpm, spo2, stage

cycle = CycleAverager()

def save_session(surface):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"cosmos_{ts}"
    img_path = os.path.join(SAVE_DIR, base + ".png")
    pygame.image.save(surface, img_path)
    log_path = os.path.join(SAVE_DIR, base + ".json")
    with open(log_path, "w") as f:
        json.dump({"session_seconds": SESSION_SECONDS, "cycles": cycle.log}, f, indent=2)
    print(f"[Cosmos] Mentve: {img_path}")
    return img_path

def finger_to_pixel(fx, fy):
    return int(fx * WIDTH), int(fy * HEIGHT)

ORB_MARGIN     = 90
ORB_MIN_COUNT  = 2
ORB_MAX_COUNT  = 9
ORB_MIN_LIFE   = 6.0
ORB_MAX_LIFE   = 16.0
ORB_FADE_TIME  = 1.8 

class Orb:
    """Egy 'test': 2-4 koncentrikus körvonalból álló, lebegő, ki-be fakuló alakzat."""
    def __init__(self, now):
        self.x = random.uniform(ORB_MARGIN, WIDTH - ORB_MARGIN)
        self.y = random.uniform(ORB_MARGIN, HEIGHT - ORB_MARGIN)
        self.phase    = random.uniform(0, math.tau)
        self.speed    = random.uniform(0.05, 0.13)
        self.drift_r  = random.uniform(6, 20)
        self.hue_off  = random.uniform(0, 3.0)
        self.beat_off = random.uniform(-0.15, 0.15)
        self.born     = now
        self.life     = random.uniform(ORB_MIN_LIFE, ORB_MAX_LIFE)
        self.n_rings  = random.randint(2, 4)
        self.ring_gap = random.uniform(7, 13)
        self.base_r   = random.uniform(9, 22)
        self.ring_widths = [random.choice([1, 1, 2, 2, 3, 4]) for _ in range(self.n_rings)]

    def position(self, t):
        x = self.x + math.cos(t * self.speed + self.phase) * self.drift_r
        y = self.y + math.sin(t * self.speed * 0.8 + self.phase) * self.drift_r
        return x, y

    def alpha_factor(self, now):
        age = now - self.born
        if age < ORB_FADE_TIME:
            return age / ORB_FADE_TIME
        remain = self.life - age
        if remain < ORB_FADE_TIME:
            return max(0.0, remain / ORB_FADE_TIME)
        return 1.0

    def is_dead(self, now):
        return (now - self.born) >= self.life

class OrbField:
    """Kezeli a 'testek' véletlenszerű meg- és eltűnését – néha több,
    néha kevesebb van belőlük egyszerre, sosem fix a darabszám."""
    def __init__(self):
        self.orbs = []
        self.target = random.randint(ORB_MIN_COUNT, ORB_MAX_COUNT)
        self.next_retarget = time.time() + random.uniform(6.0, 12.0)

    def reset(self):
        self.orbs = []
        self.target = random.randint(ORB_MIN_COUNT, ORB_MAX_COUNT)
        self.next_retarget = time.time() + random.uniform(6.0, 12.0)

    def update(self, dt, now):
        if now > self.next_retarget:
            self.target = random.randint(ORB_MIN_COUNT, ORB_MAX_COUNT)
            self.next_retarget = now + random.uniform(6.0, 13.0)
        self.orbs = [o for o in self.orbs if not o.is_dead(now)]
        if len(self.orbs) < self.target and random.random() < dt * 0.5:
            self.orbs.append(Orb(now))

    def random_position(self, t):
        if not self.orbs:
            return CENTER
        return random.choice(self.orbs).position(t)

orb_field = OrbField()

def ring_outline(surface, center, radius, color, alpha_mult=1.0, width=2):
    """Egyetlen, lágyan derengő körvonal (nem kitöltött kör)."""
    if radius <= 1.5 or alpha_mult <= 0.02:
        return
    cx, cy = int(center[0]), int(center[1])
    for i in range(4, 0, -1):
        a = int(45 * (1 - i / 4) * alpha_mult)
        if a <= 1:
            continue
        for rr in (radius + i * 1.8, radius - i * 1.8):
            if rr > 1:
                pygame.draw.circle(surface, (*color, a), (cx, cy), int(rr), 1)
    core_a = int(200 * alpha_mult)
    if core_a > 1:
        pygame.draw.circle(surface, (*color, core_a), (cx, cy), int(radius), max(1, width))

def draw_orbs(t, heartbeat_phase, stage, alert, motion):
    eff = "alert" if alert else stage
    now = time.time()
    for orb in orb_field.orbs:
        af = orb.alpha_factor(now)
        if af <= 0.02:
            continue
        x, y = orb.position(t)
        pulse = 1.0 + 0.05 * math.sin((heartbeat_phase + orb.beat_off) * math.tau)
        jitter = math.sin(t * 1.5 + orb.hue_off) * (0.5 + min(1.0, motion / 0.08) * 1.2)
        for ring_i in range(orb.n_rings):
            r = (orb.base_r + ring_i * orb.ring_gap) * pulse + jitter
            col = palette_color(eff, t + orb.hue_off + ring_i * 0.35)
            ring_outline(fx_layer, (x, y), r, col, alpha_mult=af * (1 - ring_i * 0.12),
                         width=orb.ring_widths[ring_i])

PHASE_INTRO, PHASE_SESSION, PHASE_REVIEW = "intro", "session", "review"
phase = PHASE_INTRO
session_start = 0.0
heartbeat_phase = 0.0
frozen_frame = None
running = True

SAVE_RECT    = pygame.Rect(CENTER[0] - 230, CENTER[1] + 60, 200, 64)
DISCARD_RECT = pygame.Rect(CENTER[0] + 30,  CENTER[1] + 60, 200, 64)

def reset_session():
    global session_start, heartbeat_phase, rings, sparks, meteors, last_meteor_t
    nebula_layer.fill((0, 0, 0, 0))
    session_start = time.time()
    heartbeat_phase = 0.0
    rings, sparks, meteors = [], [], []
    last_meteor_t = 0.0
    cycle.__init__()
    orb_field.reset()

def spawn_meteor(angle, intensity, color, origin=None):
    if len(meteors) >= MAX_METEORS:
        return
    if origin is None:
        origin = orb_field.random_position(time.time())
    dist  = min(WIDTH, HEIGHT) * random.uniform(0.28, 0.42)
    speed = 300 + intensity * 4
    size  = 9 + min(14, intensity * 0.18)
    meteors.append(Meteor(origin, angle, dist, color, speed, size))

while running:
    dt = clock.tick(30) / 1000.0
    now = time.time()
    t = now

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q:
                running = False
            elif e.key == pygame.K_ESCAPE:
                if phase == PHASE_SESSION:
                    phase = PHASE_REVIEW
                    frozen_frame = screen.copy()
                else:
                    running = False
            elif phase == PHASE_INTRO and e.key == pygame.K_SPACE:
                reset_session(); phase = PHASE_SESSION
            elif phase == PHASE_REVIEW and e.key == pygame.K_s:
                save_session(frozen_frame); phase = PHASE_INTRO
            elif phase == PHASE_REVIEW and e.key == pygame.K_r:
                phase = PHASE_INTRO
        elif e.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
            if e.type == pygame.MOUSEBUTTONDOWN:
                px, py = e.pos
            else:
                px, py = finger_to_pixel(e.x, e.y)
            if phase == PHASE_INTRO:
                reset_session(); phase = PHASE_SESSION
            elif phase == PHASE_REVIEW:
                if SAVE_RECT.collidepoint(px, py):
                    save_session(frozen_frame); phase = PHASE_INTRO
                elif DISCARD_RECT.collidepoint(px, py):
                    phase = PHASE_INTRO

    fx_layer.fill((0, 0, 0, 0))
    star_layer.fill((0, 0, 0, 0))
    nebula_layer.blit(NEBULA_FADE, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    for s in STARS:
        s.draw(t)

    screen.blit(BG_GRADIENT, (0, 0))
    screen.blit(nebula_layer, (0, 0))
    screen.blit(star_layer, (0, 0))

    if phase == PHASE_INTRO:
        title = FONT_BIG.render("SleepArt · Cosmos", True, (210, 215, 255))
        screen.blit(title, (CENTER[0] - title.get_width() // 2, CENTER[1] - 80))
        sub = FONT_MED.render("Tap or press SPACE to begin", True, (150, 160, 210))
        screen.blit(sub, (CENTER[0] - sub.get_width() // 2, CENTER[1] - 10))
        glow(screen, CENTER, 90 + 14 * math.sin(t * 1.3), (140, 150, 255), layers=6)

    elif phase == PHASE_SESSION:
        elapsed = now - session_start

        p = get_latest()
        motion_inst = accel_horizontal(p)
        gmag        = gyro_magnitude(p)
        bpm_inst    = compute_bpm()
        spo2_inst   = compute_spo2()
        cycle.add_sample(motion_inst, bpm_inst, spo2_inst)
        cycle.maybe_roll(elapsed)
        motion, bpm, spo2, stage = cycle.smoothed()

        alert = spo2 < 94
        eff_stage = "alert" if alert else stage
        color = palette_color(eff_stage, t)

        orb_field.update(dt, now)

        heartbeat_phase += dt * (bpm / 60.0)
        if heartbeat_phase >= 1.0:
            heartbeat_phase -= 1.0
            src = orb_field.random_position(t)
            rings.append(Ring(src, brighten(color, 1.15),
                               max_r=120,
                               speed=80 if not alert else 115))

        if gmag > max(TURN_GYRO_MIN, TURN_GYRO_MULT * 6.0) and (now - last_meteor_t) > METEOR_COOLDOWN:
            last_meteor_t = now
            angle = math.atan2(p.get("ay", 0), p.get("ax", 0)) if motion_inst > 0.02 else random.uniform(0, math.tau)
            spawn_meteor(angle, gmag, brighten(color, 1.2))

        for m in meteors:
            m.update(dt)
            m.draw(fx_layer)
        for m in meteors:
            if m.exploded:
                add_nebula_bloom((m.x, m.y), color, radius=55)
                for _ in range(16):
                    sparks.append(Spark((m.x, m.y), brighten(color, 1.15)))
        meteors = [m for m in meteors if not m.exploded]

        for r in rings:
            r.update(dt); r.draw(fx_layer)
        rings = [r for r in rings if r.alive]

        for s in sparks:
            s.update(dt); s.draw(fx_layer)
        sparks = [s for s in sparks if s.alive]

        draw_orbs(t, heartbeat_phase, stage, alert, motion)

        screen.blit(fx_layer, (0, 0))

        remain = max(0, SESSION_SECONDS - elapsed)
        mm, ss = int(remain // 60), int(remain % 60)
        hud = FONT_SMALL.render(f"{mm}:{ss:02d}   {stage}   BPM~{int(bpm)}   SpO2~{int(spo2)}%   |  Analyzer:{FLASK_PORT}",
                                 True, (170, 175, 210))
        hud.set_alpha(150)
        screen.blit(hud, (16, 14))
        pygame.draw.rect(screen, (255, 255, 255, 40), (0, HEIGHT - 4, WIDTH, 4))
        pygame.draw.rect(screen, (*brighten(color, 1.3), 200),
                          (0, HEIGHT - 4, int(WIDTH * min(1.0, elapsed / SESSION_SECONDS)), 4))

        if elapsed >= SESSION_SECONDS:
            phase = PHASE_REVIEW
            frozen_frame = screen.copy()
            print("[Cosmos] Session vége → review")

    elif phase == PHASE_REVIEW:
        screen.blit(frozen_frame, (0, 0))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((6, 4, 16, 150))
        screen.blit(overlay, (0, 0))

        title = FONT_BIG.render("Save this night?", True, (225, 225, 255))
        screen.blit(title, (CENTER[0] - title.get_width() // 2, CENTER[1] - 110))

        pygame.draw.rect(screen, (50, 150, 110), SAVE_RECT, border_radius=14)
        pygame.draw.rect(screen, (120, 230, 180), SAVE_RECT, 2, border_radius=14)
        st = FONT_MED.render("Save  [S]", True, (255, 255, 255))
        screen.blit(st, (SAVE_RECT.centerx - st.get_width() // 2, SAVE_RECT.centery - st.get_height() // 2))

        pygame.draw.rect(screen, (160, 60, 90), DISCARD_RECT, border_radius=14)
        pygame.draw.rect(screen, (235, 130, 160), DISCARD_RECT, 2, border_radius=14)
        dt_ = FONT_MED.render("Discard  [R]", True, (255, 255, 255))
        screen.blit(dt_, (DISCARD_RECT.centerx - dt_.get_width() // 2, DISCARD_RECT.centery - dt_.get_height() // 2))

    pygame.display.flip()

pygame.quit()
print("[Cosmos] Kilépés.")
