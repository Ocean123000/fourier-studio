"""平滑周期漂移和渐隐尾迹；不改变输入轮廓与原有 FFT 系数生成逻辑。"""
from __future__ import annotations

import math
import threading
import numpy as np
import cv2


def random_model(samples=2000, count=60, seed=42, max_frequency=64, spectral_decay=1.3):
    """只随机生成一次初始频谱，之后的运动是确定、平滑、可复现的。"""
    from main import FourierModel
    if not 16 <= samples or not 1 <= count <= samples:
        raise ValueError("采样数至少 16，圆盘数量不能超过采样数。")
    if max_frequency < 1 or not 0.3 <= spectral_decay <= 3:
        raise ValueError("随机频率至少 1，频谱衰减范围 0.3～3。")
    maximum = min(int(max_frequency), (samples - 1) // 2)
    pool = np.r_[np.arange(1, maximum + 1), -np.arange(1, maximum + 1)]
    actual_count = min(count, len(pool))
    rng = np.random.default_rng(seed)
    # 保留基本正、负频率，其余频率和相位由种子决定。
    fundamental = np.array([1, -1])[:actual_count]
    others = pool[~np.isin(pool, fundamental)]
    frequencies = np.r_[fundamental, rng.choice(others, actual_count - len(fundamental), replace=False)]
    radii = (1 + np.abs(frequencies)) ** (-spectral_decay) * rng.uniform(.65, 1.35, actual_count)
    coefficients = np.zeros(samples, dtype=np.complex128)
    coefficients[frequencies % samples] = radii * np.exp(1j * rng.uniform(-np.pi, np.pi, actual_count))
    z = np.fft.ifft(coefficients * samples)
    extent = max(float(np.ptp(z.real)), float(np.ptp(z.imag)))
    z *= 2 / extent
    return FourierModel(z, actual_count)


class DriftingModel:
    """给固定 FourierModel 加动态外壳；时间参数 u 为整个视频的归一化时间。"""
    def __init__(self, base, loop_seconds=60.0, drift_strength=.18,
                 drift_period=30.0, rotation_period=10.0):
        if not math.isfinite(loop_seconds) or loop_seconds <= 0:
            raise ValueError("循环时长必须为正数。")
        if not math.isfinite(drift_strength) or not 0 <= drift_strength <= .75:
            raise ValueError("drift_strength 范围 0～0.75。")
        if any(not math.isfinite(v) or v <= 0 for v in (drift_period, rotation_period)):
            raise ValueError("漂移周期和旋转周期必须为正数。")
        self.base, self.loop_seconds, self.strength = base, loop_seconds, drift_strength
        # 基础旋转必须在视频内完成整数圈；漂移也必须是整数谐波。
        self.cycles = max(1, round(loop_seconds / rotation_period))
        harmonic = max(1, round(loop_seconds / drift_period))
        i = np.arange(len(base.radii))
        self.hr = harmonic + i % 3
        self.hp = harmonic + (i + 1) % 3
        self.hf = harmonic + (i + 2) % 3
        self.offset_r = i * 2.399963229728653
        self.offset_p = i * 1.618033988749895 + .4
        self.offset_f = i * .7548776662466927 + .7
        self.weight_r = .55 + .45 * (.5 + .5 * np.sin(i * 1.7))
        self.weight_p = .55 + .45 * (.5 + .5 * np.cos(i * 1.3))
        self.weight_f = .55 + .45 * (.5 + .5 * np.sin(i * 1.1 + .5))
        self.frequency_depth = .25 * self.strength * self.weight_f
        self.frequencies = base.frequencies * self.cycles
        self.radii, self.phases = base.radii, base.phases
        self.epicycles = base.epicycles
        self.actual_periods = tuple(loop_seconds / h for h in (harmonic, harmonic + 1, harmonic + 2))

    def _state(self, times):
        u = np.mod(np.asarray(times, dtype=np.float64).reshape(-1, 1), 1.0)
        wave_r = .5 * (np.sin(2 * np.pi * u * self.hr + self.offset_r) - np.sin(self.offset_r))
        wave_p = .5 * (np.sin(2 * np.pi * u * self.hp + self.offset_p) - np.sin(self.offset_p))
        radii = self.radii * (1 + self.strength * self.weight_r * wave_r)
        phases = self.phases + 1.4 * self.strength * self.weight_p * wave_p
        frequency_hz = self.frequencies / self.loop_seconds * (
            1 + self.frequency_depth * np.sin(2 * np.pi * u * self.hf + self.offset_f))
        # 解析积分瞬时频率：不能直接使用 frequency(t) * t。
        frequency_integral = self.frequencies * self.frequency_depth / self.hf * (
            np.cos(self.offset_f) - np.cos(2 * np.pi * u * self.hf + self.offset_f))
        angles = phases + 2 * np.pi * u * self.frequencies + frequency_integral
        return radii, phases, frequency_hz, angles

    def parameters(self, u):
        """返回动态半径、相位、瞬时频率（Hz）。"""
        r, p, f, _ = self._state([u])
        return r[0], p[0], f[0]

    def angles(self, u):
        return self._state([u])[3][0]

    def chain(self, u):
        r, _, _, angles = self._state([u])
        return np.r_[0j, np.cumsum(r[0] * np.exp(1j * angles[0]))]

    def curve(self, times):
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        result = np.empty(len(times), dtype=np.complex128)
        for start in range(0, len(times), 512):
            r, _, _, angles = self._state(times[start:start + 512])
            result[start:start + len(r)] = (r * np.exp(1j * angles)).sum(axis=1)
        return result

    def bounds(self):
        # 半径变化和相位/频率变化同时计入导数上界，保证镜头覆盖整个漂移周期。
        radial_speed = self.radii * self.strength * self.weight_r * np.pi * self.hr
        angular_speed = 2 * np.pi * np.abs(self.frequencies) * (1 + self.frequency_depth)
        angular_speed += 1.4 * self.strength * self.weight_p * np.pi * self.hp
        vector_speed = radial_speed + self.radii * (1 + self.strength) * angular_speed
        center_speed = np.r_[0.0, np.cumsum(vector_speed)[:-1]]
        n = max(2048, min(16384, 4 * int(np.abs(self.frequencies).max()) + 4))
        pad = (center_speed + radial_speed) / (2 * n)
        limits = [math.inf, -math.inf, math.inf, -math.inf]
        for start in range(0, n, 256):
            u = np.arange(start, min(start + 256, n)) / n
            r, _, _, angles = self._state(u)
            v = r * np.exp(1j * angles)
            centers = np.column_stack([np.zeros(len(u)), np.cumsum(v, axis=1)])[:, :-1]
            limits[0] = min(limits[0], float(np.min(centers.real - r - pad)))
            limits[1] = max(limits[1], float(np.max(centers.real + r + pad)))
            limits[2] = min(limits[2], float(np.min(centers.imag - r - pad)))
            limits[3] = max(limits[3], float(np.max(centers.imag + r + pad)))
        return tuple(limits)


class FadingTrail:
    """按真实时间年龄生成尾迹，负时间自动读上一周期，不依赖帧调用顺序。"""
    def __init__(self, motion, trail_length=8.0, fade_power=2.0, glow_strength=.8,
                 sample_rate=120, color="cyan", constant_width=False):
        if not math.isfinite(trail_length) or not 0 < trail_length <= motion.loop_seconds:
            raise ValueError("尾迹保留时间必须大于 0 且不超过完整循环时长。")
        if not math.isfinite(fade_power) or not .2 <= fade_power <= 8:
            raise ValueError("fade_power 范围 0.2～8。")
        if not math.isfinite(glow_strength) or not 0 <= glow_strength <= 3:
            raise ValueError("glow_strength 范围 0～3。")
        if color not in {"cyan", "blue", "purple"}:
            raise ValueError("尾迹颜色必须是 cyan、blue 或 purple。")
        self.motion, self.length, self.power, self.glow = motion, trail_length, fade_power, glow_strength
        self.color = color
        self.constant_width = constant_width
        rate = max(sample_rate, 8 * float(np.abs(motion.frequencies).max())
                   * getattr(motion, "max_phase_rate", 1) / motion.loop_seconds)
        self.cache_count = max(2048, min(200000, math.ceil(motion.loop_seconds * rate)))
        self.cache = motion.curve(np.linspace(0, 1, self.cache_count + 1))
        self.cache[-1] = self.cache[0]
        segments = max(64, min(2400, math.ceil(trail_length * rate)))
        self.ages = np.linspace(trail_length, 0, segments + 1)
        ratios = (self.ages[:-1] + self.ages[1:]) / (2 * trail_length)
        self.alpha = self.opacity(ratios)
        # 连续发光线芯：保留饱和颜色，只有最新部分逐渐趋向白色。
        palette = {"cyan": (255, 235, 80), "blue": (255, 155, 65), "purple": (255, 115, 210)}
        bright = np.asarray(palette[color], dtype=float)
        self.bright = bright.astype(np.float32)
        # 16-bit 光强查表避免每帧在整片区域计算幂函数和三个颜色通道。
        levels = np.linspace(0, 1, 65536, dtype=np.float32)
        hot = levels ** 12
        tint = self.bright + .72 * hot[:, None] * (255 - self.bright)
        display = np.where(levels <= .0031308, 12.92 * levels,
                           1.055 * levels ** (1 / 2.4) - .055)
        self.light_lut = np.clip(display[:, None] * tint, 0, 255).astype(np.uint8)
        self.colors = bright[None, :] * self.alpha[:, None]
        self.width_fraction = (1 - ratios) ** .7

    def opacity(self, age_ratio):
        return np.clip(1 - np.asarray(age_ratio), 0, 1) ** self.power

    def points(self, u):
        normalized = np.mod((u % 1.0) - self.ages / self.motion.loop_seconds, 1.0)
        indices = normalized * self.cache_count
        # 接近周期边界时浮点取模/乘法可能舍入到 cache_count。
        # 使用最后一段插值到重复的首点，避免 left+1 越界。
        left = np.minimum(np.floor(indices).astype(int), self.cache_count - 1)
        fraction = indices - left
        points = self.cache[left] * (1 - fraction) + self.cache[left + 1] * fraction
        # 最新点直接取真实圆盘末端，避免缓存插值使线头与白色光点分离。
        points[-1] = self.motion.chain(u)[-1]
        return points

    def emission(self, renderer, u, with_glow=False, points=None, region=None):
        """连续线宽的发光场；每个像素取最亮一次，避免线段接头反复覆盖。"""
        if points is None:
            points = renderer.pixels(self.points(u)).astype(np.float32)
        width, height = renderer.width, renderer.height
        if region is not None:
            x0, y0, x1, y1 = region
            points = points - np.array([x0, y0], dtype=np.float32)
            width, height = x1-x0, y1-y0
        starts, ends = points[:-1], points[1:]
        lengths = np.linalg.norm(ends - starts, axis=1)
        # 分成不超过两像素的小段，只处理每段附近的像素，控制 1080p 渲染开销。
        counts = np.maximum(1, np.ceil(lengths / 2).astype(int))
        if hasattr(self.motion, "visible_between"):
            counts = np.where(self.motion.visible_between(
                u-self.ages[:-1]/self.motion.loop_seconds,
                u-self.ages[1:]/self.motion.loop_seconds), counts, 0)
        segment = np.repeat(np.arange(len(counts)), counts)
        first = np.repeat(np.cumsum(counts) - counts, counts)
        fraction0 = (np.arange(len(segment)) - first).astype(np.float32) / counts[segment].astype(np.float32)
        fraction1 = fraction0 + 1 / counts[segment].astype(np.float32)
        vectors = ends[segment] - starts[segment]
        start = starts[segment] + vectors * fraction0[:, None]
        end = starts[segment] + vectors * fraction1[:, None]
        ages = (self.ages / self.length).astype(np.float32)
        age0 = ages[segment] + (ages[segment + 1] - ages[segment]) * fraction0
        age1 = ages[segment] + (ages[segment + 1] - ages[segment]) * fraction1
        field = np.zeros((height, width), np.float32)
        flat = field.reshape(-1)
        # signed-distance 抗锯齿直接使用浮点半径，不把线宽向上取整。
        sigma = max(.8, renderer.height / 540 * (.8 + .5 * self.glow))
        glow_gain = 1 - math.exp(-self.glow) if with_glow else 0
        extent = math.ceil(max((.6 + renderer.line_width) / 2 + 2,
                               3 * sigma + 2 if glow_gain else 0))
        oy, ox = np.mgrid[-extent:extent + 1, -extent:extent + 1]
        ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        for offset in range(0, len(start), 2048):
            sl = slice(offset, offset + 2048)
            a, b = start[sl], end[sl]
            mid = np.floor((a + b) * .5).astype(int)
            x, y = mid[:, 0, None] + ox, mid[:, 1, None] + oy
            vx, vy = (b - a).T
            dx, dy = x.astype(np.float32) - a[:, 0, None], y.astype(np.float32) - a[:, 1, None]
            norm = np.maximum(vx * vx + vy * vy, 1e-12)
            f = np.clip((dx * vx[:, None] + dy * vy[:, None]) / norm[:, None], 0, 1)
            distance = np.hypot(dx - f * vx[:, None], dy - f * vy[:, None])
            age = age0[sl, None] + f * (age1[sl] - age0[sl])[:, None]
            youth = np.clip(1 - age, 0, 1)
            width_factor = 1. if self.constant_width else youth ** .7
            radius = (.6 + renderer.line_width * width_factor) / 2
            coverage = np.clip(radius + .5 - distance, 0, 1)
            if glow_gain:
                # 对每段直接计算光晕并取最大值，不因曲线慢行、折返或采样密度变亮。
                coverage = np.maximum(coverage, glow_gain * np.exp(-.5 * (distance / sigma) ** 2))
            value = (youth ** self.power * coverage).astype(np.float32)
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            # 接头、自交、密集采样均不叠加；后来的高亮不会被下一段的暗边覆盖。
            np.maximum.at(flat, (y[inside] * width + x[inside]), value[inside])
        return field

    def layer(self, renderer, u):
        # 同一个连续年龄场驱动线芯和近场光晕，沿线亮度不会被 glow 再次抬平。
        points = renderer.pixels(self.points(u)).astype(np.float32)
        # 只计算曲线附近的矩形区域，黑色背景不参与幂函数和 RGB 运算。
        # 偶数边界保持半分辨率光晕的像素对齐，边缘留出高斯核余量。
        sigma_near = max(.8, renderer.height / 540 * (.8 + .5 * self.glow))
        sigma_far = max(1.8, renderer.height / 180)
        margin = math.ceil(max(3 * sigma_near + 3, 8 * sigma_far + 3))
        lower = np.floor(points.min(axis=0) - margin).astype(int)
        upper = np.ceil(points.max(axis=0) + margin + 1).astype(int)
        x0, y0 = max(0, lower[0] // 2 * 2), max(0, lower[1] // 2 * 2)
        x1, y1 = min(renderer.width, (upper[0]+1)//2*2), min(renderer.height, (upper[1]+1)//2*2)
        output = np.zeros_like(renderer.background)
        if x1 <= x0 or y1 <= y0:
            return output
        region = (x0, y0, x1, y1)
        light = self.emission(renderer, u, with_glow=True, points=points, region=region)
        if self.glow > 0:
            small = cv2.resize(light, (max(1, (x1-x0) // 2), max(1, (y1-y0) // 2)),
                               interpolation=cv2.INTER_AREA)
            far = cv2.GaussianBlur(small, (0, 0), max(1.8, renderer.height / 180))
            halo = cv2.resize(far * .28 * (1 - math.exp(-self.glow)),
                              (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
            light = np.maximum(light, halo)
        # 查表只改计算方式，保留连续年龄场、亮白线芯及 sRGB 亮度映射。
        indices = np.clip(np.rint(light * 65535), 0, 65535).astype(np.uint16)
        output[y0:y1, x0:x1] = self.light_lut[indices]
        return output


class ExportCancelled(RuntimeError):
    """用户取消导出时结束工作线程，临时视频由原编码器负责清理。"""


class CancelToken:
    def __init__(self):
        self.event = threading.Event()

    def cancel(self):
        self.event.set()
