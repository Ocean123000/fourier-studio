"""Turn a locally installed running-script glyph into one continuous pen trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import urllib.parse
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, vectorized so no extra image package is needed."""
    pixels = np.pad(mask.astype(bool), 1)
    while True:
        deleted = 0
        for stage in (0, 1):
            p2, p3, p4 = pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:]
            p5, p6, p7 = pixels[2:, 2:], pixels[2:, 1:-1], pixels[2:, :-2]
            p8, p9 = pixels[1:-1, :-2], pixels[:-2, :-2]
            neighbors = p2.astype(np.uint8) + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = ((~p2 & p3).astype(np.uint8) + (~p3 & p4) + (~p4 & p5)
                           + (~p5 & p6) + (~p6 & p7) + (~p7 & p8) + (~p8 & p9)
                           + (~p9 & p2))
            if stage == 0:
                keep = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                keep = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            remove = (pixels[1:-1, 1:-1] & (neighbors >= 2) & (neighbors <= 6)
                      & (transitions == 1) & keep)
            deleted += int(np.count_nonzero(remove))
            pixels[1:-1, 1:-1][remove] = False
        if not deleted:
            return pixels[1:-1, 1:-1]


ROOT = Path(__file__).resolve().parent
FONT_PRESETS = {
    "华文行楷（清晰）": Path("C:/Windows/Fonts/STXINGKA.TTF"),
    "志莽行书（连笔）": ROOT / "assets/fonts/ZhiMangXing-Regular.ttf",
    "龙藏体（草书）": ROOT / "assets/fonts/LongCang-Regular.ttf",
    "刘建毛草（草书）": ROOT / "assets/fonts/LiuJianMaoCao-Regular.ttf",
    "方正舒体（签名感）": Path("C:/Windows/Fonts/FZSTK.TTF"),
    "楷体（端正）": Path("C:/Windows/Fonts/simkai.ttf"),
}


@dataclass
class Segment:
    points: np.ndarray
    kind: str
    char: int = -1
    stroke: int = -1


@dataclass
class SignaturePath:
    segments: list[Segment]
    mask: np.ndarray
    calibrated: bool = False
    source_kind: str = "font"

    @property
    def points(self):
        return np.vstack([s.points for s in self.segments])

    @property
    def boundaries(self):
        lengths = [np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum()
                   for s in self.segments]
        arc = np.r_[0., np.cumsum(lengths)]
        return arc / arc[-1]

def stroke_medians(char):
    file = ROOT / "assets/strokes" / f"{char}.json"
    if not file.is_file():
        url = "https://cdn.jsdelivr.net/npm/hanzi-writer-data@2.0/" + urllib.parse.quote(char) + ".json"
        try:
            with urllib.request.urlopen(url, timeout=6) as response:
                data = response.read().decode("utf-8")
            decoded = json.loads(data)
            if not decoded.get("medians"):
                raise ValueError("缺少笔顺")
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(data, encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"无法获取“{char}”的笔顺，请检查网络或换用已缓存的汉字。") from exc
    return [np.array([(x, 900-y) for x, y in stroke], dtype=float)
            for stroke in json.loads(file.read_text(encoding="utf-8"))["medians"]]


def _dense(points, step=1.):
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] < 1e-9:
        return np.repeat(points[:1], 2, axis=0)
    t = np.linspace(0, lengths[-1], max(2, int(lengths[-1]/step)+1))
    return np.column_stack([np.interp(t, lengths, points[:, i]) for i in (0, 1)])


def _guided_strokes(skeleton, guides):
    """Follow only the ordered main strokes, without inventing residual strokes."""
    coords = np.argwhere(skeleton)[:, ::-1].astype(float)
    lookup = {tuple(p.astype(int)): i for i, p in enumerate(coords)}
    graph = [[] for _ in coords]
    for i, (x, y) in enumerate(coords.astype(int)):
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            j = lookup.get((x+dx, y+dy))
            if j is not None:
                graph[i].append((j, np.hypot(dx,dy)))
    strokes = []
    for guide in guides:
        dense = _dense(guide, 2)
        proximity = np.min(np.linalg.norm(coords[:,None]-dense[None,:], axis=2), axis=1)
        waypoints = [int(np.argmin(np.linalg.norm(coords-p, axis=1))) for p in guide]
        parts, path = [], [waypoints[0]]
        for start, goal in zip(waypoints, waypoints[1:]):
            heap = [(0., start)]
            costs, parents = {start: 0.}, {}
            while heap:
                cost, node = heapq.heappop(heap)
                if cost > costs[node]:
                    continue
                if node == goal:
                    break
                for neighbor, length in graph[node]:
                    candidate = cost + length * (1 + (proximity[neighbor]/5)**2)
                    if candidate < costs.get(neighbor, float("inf")):
                        costs[neighbor] = candidate
                        parents[neighbor] = node
                        heapq.heappush(heap, (candidate, neighbor))
            if goal not in costs:
                if len(path) > 1:
                    parts.append(coords[path])
                path = [goal]
                continue
            chain, node = [goal], goal
            while node != start:
                node = parents[node]
                chain.append(node)
            path.extend(chain[-2::-1])
        if len(path) > 1:
            parts.append(coords[path])
        if not parts:
            # An isolated dot still needs a short movement on its actual ink.
            nearest = np.argsort(proximity)[:2]
            parts = [coords[nearest]]
        # Remove accidental loops caused by waypoints snapping to the wrong
        # arm of a junction. Keep the ordered stroke's first/last ink points.
        simple = []
        for part in parts:
            nodes, positions = [], {}
            for point in part:
                key = tuple(point)
                if key in positions:
                    cut = positions[key]
                    for old in nodes[cut+1:]:
                        positions.pop(tuple(old))
                    nodes = nodes[:cut+1]
                else:
                    positions[key] = len(nodes)
                    nodes.append(point)
            simple.append(np.asarray(nodes if len(nodes)>1 else [nodes[0], nodes[0]]))
        strokes.append(simple)
    return strokes


def _occupied(points, mask):
    p = np.rint(points).astype(int)
    valid = (p[:,0]>=0)&(p[:,0]<mask.shape[1])&(p[:,1]>=0)&(p[:,1]<mask.shape[0])
    result = np.zeros(len(p), bool)
    result[valid] = mask[p[valid,1],p[valid,0]]
    return result


def _round_path(vertices, radius):
    result = [vertices[:1]]
    last = vertices[0]
    for before, corner, after in zip(vertices, vertices[1:], vertices[2:]):
        a, b = corner-before, after-corner
        r = min(radius, np.linalg.norm(a)*.3, np.linalg.norm(b)*.3)
        entry = corner-a/max(np.linalg.norm(a),1e-9)*r
        exit = corner+b/max(np.linalg.norm(b),1e-9)*r
        result.append(_dense(np.array([last,entry])))
        t = np.linspace(0,1,max(8,int(2*r)))[:,None]
        result.append((1-t)**2*entry+2*(1-t)*t*corner+t*t*exit)
        last = exit
    result.append(_dense(np.array([last,vertices[-1]])))
    return np.vstack(result)


def outside_return(last, first, mask):
    """Round an exterior perimeter route; reject any middle collision with ink."""
    yy, xx = np.nonzero(mask)
    # A pair of cubic arcs makes a simple exterior return below the lettering.
    # Check the actual ink rather than trusting a bounding-box sketch.
    def cubic(a,b,c,d):
        t=np.linspace(0,1,1000)[:,None]
        return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d

    for margin in (55,85,125):
        mid=np.array([xx.min()-margin,yy.max()+margin],float)
        curve=np.vstack([
            cubic(last,last+[margin,margin],np.array([xx.min()+margin,yy.max()+2*margin]),mid),
            cubic(mid,mid+[-margin,-margin],np.array([first[0],yy.min()-margin]),first)[1:]])
        hit=_occupied(curve,mask)
        free=np.flatnonzero(~hit)
        if len(free) and not hit[free[0]:free[-1]+1].any():
            return curve
    low, high = np.array([xx.min(), yy.min()])-48, np.array([xx.max(), yy.max()])+48
    corners = np.array([low,[high[0],low[1]],high,[low[0],high[1]]],float)
    w,h = high-low
    perimeter = 2*(w+h)

    def exits(point):
        options=[]
        for angle in np.linspace(0,2*np.pi,128,endpoint=False):
            direction=np.array([np.cos(angle),np.sin(angle)])
            candidates=[(bound[i]-point[i])/direction[i] for i in (0,1)
                        for bound in (low,high) if abs(direction[i])>1e-9
                        and (bound[i]-point[i])/direction[i]>0]
            distance=min(candidates)
            end=point+distance*direction
            occupied=_occupied(_dense(np.array([point,end])),mask)
            # Only the contiguous cap containing the endpoint may be crossed.
            free=np.flatnonzero(~occupied)
            if not len(free) or occupied[free[0]:].any():
                continue
            if abs(end[1]-low[1])<1e-4: pos=end[0]-low[0]
            elif abs(end[0]-high[0])<1e-4: pos=w+end[1]-low[1]
            elif abs(end[1]-high[1])<1e-4: pos=w+h+high[0]-end[0]
            else: pos=2*w+h+high[1]-end[1]
            options.append((end,pos,distance))
        return options

    candidates=[]
    corner_pos=np.array([0,w,w+h,2*w+h],float)
    for end,p,d1 in exits(last):
        for entry,q,d2 in exits(first):
            for direction in (1,-1):
                span=((q-p)*direction)%perimeter
                positions=((corner_pos-p)*direction)%perimeter
                intermediate=corners[np.argsort(positions)[positions[np.argsort(positions)]<span]]
                vertices=np.vstack([last,end,intermediate,entry,first])
                candidates.append((d1+d2+span,vertices))
    for _,vertices in sorted(candidates,key=lambda c:c[0]):
        for radius in (22,10,3):
            curve=_round_path(vertices,radius)
            hit=_occupied(curve,mask)
            free=np.flatnonzero(~hit)
            if len(free) and not hit[free[0]:free[-1]+1].any():
                return curve
    raise ValueError("找不到不穿过字形的首尾回线，请更换字体。")


def build_ordered_signature(text: str, close_loop=True) -> SignaturePath:
    """Use ordered medians directly, with one scale and baseline for all glyphs.

    This is standard single-line lettering, not a running-script conversion.
    No font skeleton snapping, per-glyph stretching or additional branches.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        raise ValueError("请输入要生成签名的文字。")
    strokes = []
    scale, advance = 240/1024, 260.
    for index, char in enumerate(chars):
        medians = stroke_medians(char)
        if not medians:
            raise ValueError(f"“{char}”没有可用的标准笔顺数据。")
        for order, median in enumerate(medians):
            if median.ndim!=2 or median.shape[1]!=2 or len(median)<2 or not np.isfinite(median).all():
                raise ValueError(f"“{char}”的第 {order+1} 笔数据无效。")
            strokes.append(Segment(median*scale+[index*advance, 0], "stroke", index, order))
    points = np.vstack([s.points for s in strokes])
    offset = 60-points.min(axis=0)
    for stroke in strokes:
        stroke.points = stroke.points+offset
    maximum = np.vstack([s.points for s in strokes]).max(axis=0)
    mask = np.zeros((int(np.ceil(maximum[1]))+61, int(np.ceil(maximum[0]))+61), np.uint8)
    segments = []
    for stroke in strokes:
        cv2.polylines(mask, [np.rint(stroke.points).astype(np.int32)], False, 1, 5)
        if segments:
            kind = "char" if segments[-1].char!=stroke.char else "lift"
            segments.append(Segment(np.array([segments[-1].points[-1], stroke.points[0]]),
                                    kind, stroke.char, stroke.stroke))
        segments.append(stroke)
    mask = mask.astype(bool)
    closure = (outside_return(segments[-1].points[-1], segments[0].points[0], mask)
               if close_loop else np.array([segments[-1].points[-1], segments[0].points[0]]))
    segments.append(Segment(closure, "return" if close_loop else "lift"))
    return SignaturePath(segments, mask, source_kind="ordered")


def build_signature(text: str, font_path: str | Path, close_loop=True) -> SignaturePath:
    chars=[c for c in text if not c.isspace()]
    if not chars:
        raise ValueError("请输入要生成签名的文字。")
    font=ImageFont.truetype(str(font_path),240)
    boxes=[font.getbbox(c) for c in chars]
    advances=[font.getlength(c)+20 for c in chars]
    top=min(b[1] for b in boxes)
    mask=np.zeros((max(b[3] for b in boxes)-top+120,int(sum(advances))+120),bool)
    segments=[]
    left=60.
    anchors_file=ROOT/'assets/strokes/font_anchors.json'
    anchors=json.loads(anchors_file.read_text(encoding='utf-8')) if anchors_file.is_file() else {}
    calibration_file = ROOT / "assets/signatures/calibration.json"
    calibration = None
    if calibration_file.is_file():
        saved = json.loads(calibration_file.read_text(encoding="utf-8"))
        if hashlib.sha256(Path(font_path).read_bytes()).hexdigest() == saved["font_sha256"]:
            calibration = saved["glyphs"]
    for index,(char,box,advance) in enumerate(zip(chars,boxes,advances)):
        if bytes(font.getmask(char)) == bytes(font.getmask(chr(0x10FFFF))):
            raise ValueError(f"当前字体不包含“{char}”，请选择另一款字体。")
        width,height=box[2]-box[0],box[3]-box[1]
        glyph=Image.new('L',(width+40,height+40))
        ImageDraw.Draw(glyph).text((20-box[0],20-box[1]),char,font=font,fill=255)
        ink=np.asarray(glyph)>=128
        if not ink.any():
            raise ValueError(f"字体不包含“{char}”。")
        guides=stroke_medians(char)
        key=Path(font_path).name.lower()
        if key in anchors and char in anchors[key]:
            guides=[np.array(p,float)-[0,10] for p in anchors[key][char]]
        else:
            all_points=np.vstack(guides)
            yy,xx=np.nonzero(ink)
            extent=np.ptp(all_points,axis=0)
            guides=[(p-all_points.min(axis=0))/np.maximum(extent,1)*[xx.max()-xx.min(),yy.max()-yy.min()]
                    +[xx.min(),yy.min()] for p in guides]
        offset=np.array([round(left)+box[0]-20,60+box[1]-top-20])
        y,x=int(offset[1]),int(offset[0])
        mask[y:y+ink.shape[0],x:x+ink.shape[1]]|=ink
        strokes = ([[np.asarray(p, dtype=float)] for p in calibration[char]] if calibration and char in calibration
                   else _guided_strokes(_thin(ink), guides))
        for stroke,parts in enumerate(strokes):
            for points in parts:
                points=points+offset
                if segments:
                    kind='char' if segments[-1].char!=index else 'lift'
                    segments.append(Segment(np.array([segments[-1].points[-1],points[0]]),kind,index,stroke))
                segments.append(Segment(points,'stroke',index,stroke))
        left+=advance
    if close_loop:
        segments.append(Segment(outside_return(segments[-1].points[-1],segments[0].points[0],mask),'return'))
    else:
        segments.append(Segment(np.array([segments[-1].points[-1],segments[0].points[0]]),'lift'))
    return SignaturePath(segments, mask, calibrated=calibration is not None and all(c in calibration for c in chars))


def font_signature_trajectory(text, font_path, close_loop=True):
    return build_signature(text,font_path,close_loop).points


class SignatureMotion:
    """Fixed Fourier coefficients with one linear clock for all epicycles.

    The font supplies only the sampled target. Ink and the pen tip both follow
    the truncated Fourier sum, including all transfers and the return curve.
    """
    def __init__(self, base, path, loop_seconds=60, cycles=1, origin=0j, scale=1,
                 show_transfers=False):
        self.base, self.path, self.loop_seconds, self.cycles = base, path, loop_seconds, cycles
        self.origin, self.scale = origin, scale
        self.radii, self.phases, self.epicycles = base.radii, base.phases, base.epicycles
        self.frequencies = base.frequencies * cycles
        self.actual_periods = ()
        self.strength = 0.
        self.max_phase_rate = 1.
        self.show_transfers = show_transfers
        self.arc = path.boundaries
        self.ink_segments = np.array([show_transfers or s.kind != "lift" for s in path.segments])
        self.ink_integral = np.r_[0., np.cumsum(np.diff(self.arc)*self.ink_segments)]

    def phase(self, times):
        return np.asarray(times) * self.cycles

    def phase_rate(self, times):
        return np.full_like(np.asarray(times, dtype=float), self.cycles)

    def chain(self, t):
        return self.base.chain(float(self.phase(t)))

    def curve(self, times):
        return self.base.curve(self.phase(times))

    def angles(self, t):
        return self.base.phases + 2*np.pi*self.base.frequencies*float(self.phase(t))

    def bounds(self):
        return self.base.bounds()

    def parameters(self, t):
        return self.radii, self.phases, self.frequencies/self.loop_seconds

    def visible(self, times):
        local = np.mod(self.phase(times), 1)
        index = np.clip(np.searchsorted(self.arc, local, side="right")-1, 0, len(self.ink_segments)-1)
        return self.ink_segments[index]

    def visible_between(self, starts, ends):
        """An interval crossing a pen-up segment must not paint a diagonal."""
        if self.show_transfers:
            return np.ones_like(np.asarray(starts), dtype=bool)
        def integral(times):
            phase = self.phase(times)
            return (np.floor(phase)*self.ink_integral[-1]
                    +np.interp(np.mod(phase, 1), self.arc, self.ink_integral))/self.cycles
        duration = np.asarray(ends)-np.asarray(starts)
        return np.isclose(integral(ends)-integral(starts), duration, atol=1e-12, rtol=0)
