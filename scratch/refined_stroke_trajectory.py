"""
Refined single-line stroke trajectory with natural calligraphic air-arcs
"""
import urllib.request, urllib.parse, json
from pathlib import Path
import numpy as np
import cv2

CACHE_DIR = Path("assets/strokes")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_char_medians(char: str):
    cache_file = CACHE_DIR / f"{char}.json"
    if cache_file.is_file():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data["medians"]
    url = f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@2.0/{urllib.parse.quote(char)}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        content = resp.read().decode("utf-8")
        cache_file.write_text(content, encoding="utf-8")
        data = json.loads(content)
        return data["medians"]

def build_stroke_trajectory(chars="傅里叶", char_width=1000, char_spacing=180, speed_ratio=6.0):
    all_char_strokes = []
    for idx, char in enumerate(chars):
        medians = get_char_medians(char)
        x_offset = idx * (char_width + char_spacing)
        char_strokes = []
        for stroke in medians:
            # HanziWriter: Y=900 is top, Y=100 is bottom.
            # Convert to image Y: top is ~80, bottom is ~900
            pts = np.array([[pt[0] + x_offset, 960 - pt[1]] for pt in stroke], dtype=np.float64)
            char_strokes.append(pts)
        all_char_strokes.append(char_strokes)

    segments = [] # list of (points, is_air)

    for c_idx, char_strokes in enumerate(all_char_strokes):
        for s_idx, stroke_pts in enumerate(char_strokes):
            # Real stroke
            segments.append((stroke_pts, False))

            # If there's a next stroke in the SAME character
            if s_idx < len(char_strokes) - 1:
                p_end = stroke_pts[-1]
                p_next_start = char_strokes[s_idx + 1][0]
                # Slight air arc inside character
                dist = np.linalg.norm(p_next_start - p_end)
                if dist > 80:
                    mid = (p_end + p_next_start) / 2
                    # slight curve
                    mid_offset = np.array([0.0, 15.0])
                    arc_pts = []
                    for t in np.linspace(0, 1, 10):
                        pt = (1-t)**2 * p_end + 2*(1-t)*t * (mid + mid_offset) + t**2 * p_next_start
                        arc_pts.append(pt)
                    segments.append((np.array(arc_pts), True))
                else:
                    segments.append((np.array([p_end, p_next_start]), True))

        # If there's a NEXT character
        if c_idx < len(all_char_strokes) - 1:
            p_end_char = char_strokes[-1][-1]
            p_start_next_char = all_char_strokes[c_idx + 1][0][0]
            # Elegant inter-character calligraphic leap arc (空中飞渡)
            mid = (p_end_char + p_start_next_char) / 2
            # Sag downwards slightly like a brush flying through the air
            c_mid = np.array([mid[0], mid[1] + 60.0])
            arc_pts = []
            for t in np.linspace(0, 1, 20):
                pt = (1-t)**2 * p_end_char + 2*(1-t)*t * c_mid + t**2 * p_start_next_char
                arc_pts.append(pt)
            segments.append((np.array(arc_pts), True))

    # Return Underline: From tail of last char to head of first char
    p_tail = all_char_strokes[-1][-1][-1] # end of last stroke of last char
    p_head = all_char_strokes[0][0][0]     # start of first stroke of first char
    total_w = len(chars) * (char_width + char_spacing) - char_spacing
    bottom_y = 1060

    c1 = (p_tail[0] + 80, p_tail[1] + 120)
    c2 = (total_w + 30, bottom_y)
    c3 = (total_w // 2, bottom_y + 40)
    c4 = (-30, bottom_y)
    c5 = (-80, (p_head[1] + bottom_y) // 2)

    return_flourish = []
    for t in np.linspace(0, 1, 40):
        x = (1-t)**3 * p_tail[0] + 3*(1-t)**2*t * c1[0] + 3*(1-t)*t**2 * c2[0] + t**3 * c3[0]
        y = (1-t)**3 * p_tail[1] + 3*(1-t)**2*t * c1[1] + 3*(1-t)*t**2 * c2[1] + t**3 * c3[1]
        return_flourish.append([x, y])
    for t in np.linspace(0, 1, 40):
        x = (1-t)**3 * c3[0] + 3*(1-t)**2*t * c4[0] + 3*(1-t)*t**2 * c5[0] + t**3 * p_head[0]
        y = (1-t)**3 * c3[1] + 3*(1-t)**2*t * c4[1] + 3*(1-t)*t**2 * c5[1] + t**3 * p_head[1]
        return_flourish.append([x, y])

    # Return flourish is drawn smoothly (like a signature flourish)
    segments.append((np.array(return_flourish), False))

    # Variable Speed Parametric Sampling
    seg_times = []
    total_time = 0.0
    for pts, is_air in segments:
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = float(dists.sum())
        speed = speed_ratio if is_air else 1.0
        dt = L / speed
        # air transit should have a small floor time so it doesn't vanish
        dt = max(dt, 8.0 if is_air else 25.0)
        seg_times.append(dt)
        total_time += dt

    N_samples = 2400
    resampled = []
    for (pts, is_air), dt in zip(segments, seg_times):
        n_pts = max(2, int(round(N_samples * (dt / total_time))))
        dists = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        total_L = dists[-1]
        if total_L < 1e-6:
            interp_pts = np.tile(pts[0], (n_pts, 1))
        else:
            targets = np.linspace(0, total_L, n_pts, endpoint=False)
            interp_x = np.interp(targets, dists, pts[:, 0])
            interp_y = np.interp(targets, dists, pts[:, 1])
            interp_pts = np.column_stack([interp_x, interp_y])
        resampled.append(interp_pts)

    loop_points = np.vstack(resampled)
    return loop_points

loop_pts = build_stroke_trajectory("傅里叶")
print(f"Refined trajectory generated: {len(loop_pts)} points.")
