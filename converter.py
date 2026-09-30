#!/usr/bin/env python3
import argparse
import math
import re
import sys
import time
from collections import deque

try:
    from svgelements import SVG, Shape, Path, Move, Close
except ImportError:
    SVG = None


def load_polylines(svg_file, curve_samples=24):
    if SVG is None:
        sys.exit("Hiányzik a svgelements: pip install svgelements")

    svg = SVG.parse(svg_file, reify=True)
    polylines = []

    for el in svg.elements():
        if not isinstance(el, Shape):
            continue
        path = el if isinstance(el, Path) else Path(el)
        path.reify()

        cur = []
        for seg in path:
            if isinstance(seg, Move):
                if len(cur) > 1:
                    polylines.append(cur)
                cur = [(seg.end.x, seg.end.y)]
            elif isinstance(seg, Close):
                if cur:
                    cur.append((seg.end.x, seg.end.y))
            elif type(seg).__name__ == "Line":
                if not cur:
                    cur = [(seg.start.x, seg.start.y)]
                cur.append((seg.end.x, seg.end.y))
            else:
                if not cur:
                    cur = [(seg.start.x, seg.start.y)]
                for i in range(1, curve_samples + 1):
                    p = seg.point(i / curve_samples)
                    cur.append((p.x, p.y))
        if len(cur) > 1:
            polylines.append(cur)

    return polylines


def fit_to_bed(polys, size_mm, margin_mm):
    xs = [p[0] for pl in polys for p in pl]
    ys = [p[1] for pl in polys for p in pl]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = maxx - minx, maxy - miny
    avail = size_mm - 2 * margin_mm

    sx = avail / w if w > 0 else float("inf")
    sy = avail / h if h > 0 else float("inf")
    s = min(sx, sy)
    if math.isinf(s):
        s = 1.0

    ox = margin_mm + (avail - w * s) / 2
    oy = margin_mm + (avail - h * s) / 2

    out = []
    for pl in polys:
        out.append([((x - minx) * s + ox, (maxy - y) * s + oy) for x, y in pl])
    return out, s


def simplify(polys, min_dist):
    out = []
    for pl in polys:
        new = [pl[0]]
        for p in pl[1:-1]:
            if math.dist(p, new[-1]) >= min_dist:
                new.append(p)
        if math.dist(pl[-1], new[-1]) > 1e-6 or len(new) == 1:
            new.append(pl[-1])
        if len(new) > 1:
            out.append(new)
    return out


def optimize_order(polys):
    remaining = polys[:]
    pos = (0.0, 0.0)
    ordered = []
    while remaining:
        best_i, best_rev, best_d = 0, False, float("inf")
        for i, pl in enumerate(remaining):
            d0 = math.dist(pos, pl[0])
            d1 = math.dist(pos, pl[-1])
            if d0 < best_d:
                best_i, best_rev, best_d = i, False, d0
            if d1 < best_d:
                best_i, best_rev, best_d = i, True, d1
        pl = remaining.pop(best_i)
        if best_rev:
            pl = pl[::-1]
        ordered.append(pl)
        pos = pl[-1]
    return ordered


def generate_gcode(polys, a):
    g = ["G21", "G90"]
    if a.set_zero:
        g.append("G92 X0 Y0")

    def pen_up():
        if a.pen_mode == "servo":
            return [f"M3 S{a.servo_up}", f"G4 P{a.pen_delay}"]
        return [f"G0 Z{a.z_up}", f"G4 P{a.pen_delay}"]

    def pen_down():
        if a.pen_mode == "servo":
            return [f"M3 S{a.servo_down}", f"G4 P{a.pen_delay}"]
        return [f"G1 Z{a.z_down} F{a.plunge_feed}", f"G4 P{a.pen_delay}"]

    g += pen_up()
    for pl in polys:
        x, y = pl[0]
        g.append(f"G0 X{x:.3f} Y{y:.3f} F{a.travel_feed}")
        g += pen_down()
        first = True
        for x, y in pl[1:]:
            if first:
                g.append(f"G1 X{x:.3f} Y{y:.3f} F{a.draw_feed}")
                first = False
            else:
                g.append(f"G1 X{x:.3f} Y{y:.3f}")
        g += pen_up()

    g.append(f"G0 X0 Y0 F{a.travel_feed}")
    return g


class GrblSender:
    RX_BUFFER = 127

    def __init__(self, port, baud):
        try:
            import serial
        except ImportError:
            sys.exit("Hiányzik a pyserial: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.2)

    def wake(self):
        self.ser.write(b"\r\n\r\n")
        time.sleep(2)
        self.ser.reset_input_buffer()
        self.ser.write(b"$X\n")
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    @staticmethod
    def clean(lines):
        out = []
        for l in lines:
            l = re.sub(r"\(.*?\)", "", l.split(";")[0]).strip().upper()
            if l:
                out.append(l)
        return out

    def stream(self, raw_lines):
        lines = self.clean(raw_lines)
        total = len(lines)
        pending = deque()
        i = acked = 0
        t0 = time.time()

        while acked < total:
            while i < total and sum(pending) + len(lines[i]) + 1 <= self.RX_BUFFER - 1:
                self.ser.write((lines[i] + "\n").encode())
                pending.append(len(lines[i]) + 1)
                i += 1

            resp = self.ser.readline().decode(errors="ignore").strip()
            if not resp:
                continue
            if resp == "ok":
                pending.popleft()
                acked += 1
                if acked % 20 == 0 or acked == total:
                    print(f"\r{acked}/{total} sor ({100*acked//total}%)  "
                          f"{time.time()-t0:.0f} s", end="", flush=True)
            elif resp.startswith("error") or resp.startswith("ALARM"):
                raise RuntimeError(f"GRBL hiba a(z) {acked+1}. sornál "
                                   f"('{lines[acked]}'): {resp}")
            else:
                print(f"\n[GRBL] {resp}")
        print()
        self.wait_idle()

    def wait_idle(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.ser.write(b"?")
            resp = self.ser.readline().decode(errors="ignore")
            if "<Idle" in resp:
                return
            time.sleep(0.2)

    def emergency_stop(self, pen_up_line):
        self.ser.write(b"!")
        time.sleep(0.1)
        self.ser.write(b"\x18")
        time.sleep(1.5)
        self.ser.reset_input_buffer()
        self.ser.write((pen_up_line + "\n").encode())

    def close(self):
        self.ser.close()


def main():
    ap = argparse.ArgumentParser(description="SVG -> G-code -> GRBL plotter")
    ap.add_argument("svg", nargs="?", help="bemeneti SVG fájl")
    ap.add_argument("--send-gcode", help="kész .gcode fájl küldése SVG helyett")
    ap.add_argument("-o", "--out", default="output.gcode", help="G-code kimenet")
    ap.add_argument("--dry-run", action="store_true", help="csak generál, nem küld")

    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)

    ap.add_argument("--size", type=float, default=120.0, help="munkaterület mm (négyzet)")
    ap.add_argument("--margin", type=float, default=5.0, help="margó mm")
    ap.add_argument("--min-dist", type=float, default=0.05, help="pont-szűrés mm")
    ap.add_argument("--curve-samples", type=int, default=24)

    ap.add_argument("--draw-feed", type=int, default=1500, help="rajzolási sebesség mm/min")
    ap.add_argument("--travel-feed", type=int, default=4000, help="üres mozgás mm/min")
    ap.add_argument("--plunge-feed", type=int, default=500, help="Z süllyesztés mm/min")

    ap.add_argument("--pen-mode", choices=["servo", "z"], default="servo")
    ap.add_argument("--servo-up", type=int, default=0, help="M3 S érték felemelve")
    ap.add_argument("--servo-down", type=int, default=1000, help="M3 S érték lent")
    ap.add_argument("--z-up", type=float, default=3.0)
    ap.add_argument("--z-down", type=float, default=0.0)
    ap.add_argument("--pen-delay", type=float, default=0.15, help="várakozás tollmozgás után (s)")

    ap.add_argument("--no-zero", dest="set_zero", action="store_false",
                    help="ne állítsa 0,0-ra a jelenlegi pozíciót (G92)")
    a = ap.parse_args()

    if a.send_gcode:
        with open(a.send_gcode) as f:
            gcode = f.read().splitlines()
    else:
        if not a.svg:
            ap.error("adj meg egy SVG fájlt vagy használd a --send-gcode kapcsolót")
        polys = load_polylines(a.svg, a.curve_samples)
        if not polys:
            sys.exit("Nem találtam rajzolható útvonalat az SVG-ben "
                     "(a szöveget előbb alakítsd útvonallá!).")
        polys, scale = fit_to_bed(polys, a.size, a.margin)
        polys = simplify(polys, a.min_dist)
        polys = optimize_order(polys)
        gcode = generate_gcode(polys, a)
        with open(a.out, "w") as f:
            f.write("\n".join(gcode) + "\n")
        pts = sum(len(p) for p in polys)
        print(f"{len(polys)} vonal, {pts} pont, skála: {scale:.4f} mm/egység -> {a.out}")

    if a.dry_run:
        return

    pen_up_line = f"M3 S{a.servo_up}" if a.pen_mode == "servo" else f"G0 Z{a.z_up}"
    sender = GrblSender(a.port, a.baud)
    try:
        sender.wake()
        print("Toll a bal alsó sarokban? Indítás 3 mp múlva... (Ctrl+C = leállítás)")
        time.sleep(3)
        sender.stream(gcode)
        print("Kész.")
    except KeyboardInterrupt:
        print("\nLeállítás...")
        sender.emergency_stop(pen_up_line)
    except RuntimeError as e:
        print(f"\n{e}")
        sender.emergency_stop(pen_up_line)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
