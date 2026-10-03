"""
Single-line stroke-order calligraphy trajectory with air-transit speed control
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

# 1. Fetch medians for '傅里叶'
chars = ["刘", "鸣", "昊"]
all_strokes = []

char_width = 1024
char_spacing = 150 # gap between characters

for idx, char in enumerate(chars):
    medians = get_char_medians(char)
    x_offset = idx * (char_width + char_spacing)
    print(f"Char '{char}': {len(medians)} strokes, x_offset={x_offset}")
    for stroke in medians:
        # stroke is list of [x, y] in HanziWriter coords (0..1024, y-up where 900 is top)
        # Flip y so it's normal image coordinates (0 at top, 1024 at bottom)
        # In HanziWriter: y=900 is top, y=100 is bottom. So y_img = 950 - y
        stroke_pts = np.array([[pt[0] + x_offset, 950 - pt[1]] for pt in stroke], dtype=np.float64)
        all_strokes.append(stroke_pts)

print(f"Total strokes across all characters: {len(all_strokes)}")

# 2. Build continuous path with speed weighting
# Every segment has (points, is_air_transit)
segments = []

for i in range(len(all_strokes)):
    curr_stroke = all_strokes[i]
    segments.append((curr_stroke, False)) # real stroke
    
    if i < len(all_strokes) - 1:
        next_stroke = all_strokes[i + 1]
        p_end = curr_stroke[-1]
        p_start_next = next_stroke[0]
        # In-air transit leap between strokes
        transit_pts = np.array([p_end, p_start_next])
        segments.append((transit_pts, True)) # air transit (fast)

# 3. Closing loop: from end of last stroke of '昊' back to start of first stroke of '刘'
p_tail = all_strokes[-1][-1] # end of '昊' (捺脚)
p_head = all_strokes[0][0]   # start of '刘' (点起笔)

# Return flourish sweeping under all characters:
total_w = len(chars) * (char_width + char_spacing) - char_spacing
bottom_y = 1100

c1 = (p_tail[0] + 100, p_tail[1] + 150)
c2 = (total_w, bottom_y)
c3 = (total_w // 2, bottom_y + 40)
c4 = (0, bottom_y)
c5 = (-80, (p_head[1] + bottom_y) // 2)

return_pts = []
for t in np.linspace(0, 1, 40):
    x = (1-t)**3 * p_tail[0] + 3*(1-t)**2*t * c1[0] + 3*(1-t)*t**2 * c2[0] + t**3 * c3[0]
    y = (1-t)**3 * p_tail[1] + 3*(1-t)**2*t * c1[1] + 3*(1-t)*t**2 * c2[1] + t**3 * c3[1]
    return_pts.append([x, y])
for t in np.linspace(0, 1, 40):
    x = (1-t)**3 * c3[0] + 3*(1-t)**2*t * c4[0] + 3*(1-t)*t**2 * c5[0] + t**3 * p_head[0]
    y = (1-t)**3 * c3[1] + 3*(1-t)**2*t * c4[1] + 3*(1-t)*t**2 * c5[1] + t**3 * p_head[1]
    return_pts.append([x, y])

segments.append((np.array(return_pts), False)) # return flourish is visible

# 4. Parametric Time Sampling with Variable Speed
# Stroke writing speed: v_write = 1.0 (dense sampling, spends lots of time)
# In-air transit speed: v_air = 8.0 (fast leap, spends very little time!)
total_time = 0.0
seg_times = []

for pts, is_air in segments:
    # Arc length of this segment
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(dists.sum())
    speed = 8.0 if is_air else 1.0
    dt = L / speed
    # Ensure minimum time so it doesn't vanish
    dt = max(dt, 5.0 if is_air else 20.0)
    seg_times.append(dt)
    total_time += dt

N_samples = 2000
sampled_points = []

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
    sampled_points.append(interp_pts)

continuous_loop = np.vstack(sampled_points)
print(f"Continuous closed loop generated: {len(continuous_loop)} points.")

# Save preview
img_w = int(total_w + 300)
img_h = int(bottom_y + 150)
preview = np.zeros((img_h, img_w, 3), dtype=np.uint8)

# Draw full loop with color gradient to see stroke progression
for i in range(len(continuous_loop) - 1):
    p1 = tuple(np.int32(continuous_loop[i]))
    p2 = tuple(np.int32(continuous_loop[i + 1]))
    t = i / len(continuous_loop)
    # Yellow to Cyan to Magenta gradient
    c = (int(255 * (1 - t)), int(255 * t), 255)
    cv2.line(preview, p1, p2, c, 2, cv2.LINE_AA)

cv2.imwrite("scratch/stroke_order_preview.png", preview)
print("Saved scratch/stroke_order_preview.png")
