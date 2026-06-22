from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LESION_CHANNELS = [
    "microaneurysm",
    "hemorrhage",
    "hard_exudate",
    "cotton_wool_spot",
    "neovascularization",
]


def norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = arr - float(arr.min())
    return arr / max(float(arr.max()), 1e-8)


@dataclass(frozen=True)
class DRXAIPreprocessor:
    """Fundus preprocessing aligned with the DR-XAI fundus pipeline.

    The local DR-XAI reference uses retinal masking, CLAHE, green-channel emphasis,
    and square resize/padding. This implementation keeps the same operations local
    so the graph repository is self-contained.
    """

    image_size: int = 448
    use_mask: bool = True
    use_clahe: bool = True
    use_green: bool = True

    def make_retina_mask(self, rgb: np.ndarray) -> np.ndarray:
        green = rgb[:, :, 1]
        green_blur = cv2.medianBlur(green, 9)
        _, thresh = cv2.threshold(green_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones(green.shape, np.uint8) * 255
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(green, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle = np.zeros_like(green, np.uint8)
        cv2.circle(circle, (int(x), int(y)), int(radius * 0.98), 255, -1)
        return cv2.bitwise_and(mask, circle)

    @staticmethod
    def apply_clahe(rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        return cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    @staticmethod
    def emphasize_green(rgb: np.ndarray) -> np.ndarray:
        r_chan, g_chan, b_chan = cv2.split(rgb)
        mixed = cv2.addWeighted(r_chan, 0.1, g_chan, 0.8, 0)
        mixed = cv2.addWeighted(mixed, 1.0, b_chan, 0.1, 0)
        return cv2.merge([mixed, mixed, mixed])

    @staticmethod
    def resize_pad_square(rgb: np.ndarray, size: int) -> np.ndarray:
        height, width = rgb.shape[:2]
        scale = size / max(height, width)
        new_height, new_width = int(round(height * scale)), int(round(width * scale))
        resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pad_top = (size - new_height) // 2
        pad_bottom = size - new_height - pad_top
        pad_left = (size - new_width) // 2
        pad_right = size - new_width - pad_left
        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

    def __call__(self, path: str | Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.use_mask:
            mask = self.make_retina_mask(rgb)
            rgb = rgb.copy()
            rgb[mask == 0] = 0
        if self.use_clahe:
            rgb = self.apply_clahe(rgb)
        if self.use_green:
            rgb = self.emphasize_green(rgb)
        return self.resize_pad_square(rgb, self.image_size)


def conservative_neovascularization(vessel_dark: np.ndarray) -> np.ndarray:
    """Conservative neovascularization proxy.

    Neovascularization is rarer than other DR lesions. A broad vessel response can
    otherwise create too much false NV signal, so this channel keeps only fine,
    high-frequency, disordered vessel-like evidence and suppresses smooth vessels.
    """

    vessel_smooth = cv2.GaussianBlur(vessel_dark, (0, 0), 2.8)
    fine_vessel = norm01(np.maximum(vessel_dark - vessel_smooth, 0.0))
    lap = norm01(np.abs(cv2.Laplacian(vessel_dark, cv2.CV_32F)))
    gx = cv2.Sobel(vessel_dark, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(vessel_dark, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / np.maximum(gxx + gyy, 1e-8)
    orientation_disorder = 1.0 - np.clip(coherence, 0.0, 1.0)
    thick_vessel_suppressor = 1.0 - norm01(vessel_smooth)
    raw = (0.55 * fine_vessel + 0.45 * lap) * (0.25 + 0.75 * orientation_disorder) * thick_vessel_suppressor
    raw = norm01(raw)
    cutoff = np.quantile(raw, 0.94)
    return norm01(np.where(raw >= cutoff, raw, 0.0))


def lesion_evidence_maps(rgb: np.ndarray) -> dict[str, np.ndarray]:
    rgbf = rgb.astype(np.float32) / 255.0
    r_chan, g_chan, b_chan = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    green = rgb[..., 1]

    vessel_dark = norm01(cv2.morphologyEx(255 - green, cv2.MORPH_TOPHAT, np.ones((13, 13), np.uint8)))
    red_excess = norm01(r_chan - 0.5 * g_chan - 0.5 * b_chan)
    dark_red = norm01(red_excess * (1.0 - gray))
    yellow = norm01((r_chan + g_chan) / 2.0 - b_chan)
    bright_yellow = norm01(yellow * val * sat)
    bright_soft = norm01(val * (1.0 - sat) * norm01(gray))

    return {
        "microaneurysm": norm01(dark_red * (1.0 - cv2.GaussianBlur(dark_red, (0, 0), 2.5))),
        "hemorrhage": norm01(cv2.GaussianBlur(dark_red, (0, 0), 1.4)),
        "hard_exudate": bright_yellow,
        "cotton_wool_spot": bright_soft,
        "neovascularization": conservative_neovascularization(vessel_dark),
    }


def map_stats(arr: np.ndarray, prefix: str, threshold: float = 0.35) -> dict[str, float]:
    evidence = norm01(arr)
    mask = evidence >= threshold
    yy, xx = np.indices(evidence.shape)
    if mask.any():
        weights = evidence * mask
        wsum = float(weights.sum())
        cx = float((xx * weights).sum() / max(wsum, 1e-8) / evidence.shape[1])
        cy = float((yy * weights).sum() / max(wsum, 1e-8) / evidence.shape[0])
    else:
        cx = cy = 0.5
    hist, _ = np.histogram(evidence, bins=16, range=(0, 1), density=True)
    hist = hist / max(float(hist.sum()), 1e-8)
    return {
        f"{prefix}_mean": float(evidence.mean()),
        f"{prefix}_std": float(evidence.std()),
        f"{prefix}_max": float(evidence.max()),
        f"{prefix}_area_035": float((evidence >= 0.35).mean()),
        f"{prefix}_area_050": float((evidence >= 0.50).mean()),
        f"{prefix}_centroid_x": cx,
        f"{prefix}_centroid_y": cy,
        f"{prefix}_entropy": float(-(hist * np.log(hist + 1e-8)).sum()),
        f"{prefix}_superior_fraction": float(mask[: evidence.shape[0] // 2].mean()),
        f"{prefix}_inferior_fraction": float(mask[evidence.shape[0] // 2 :].mean()),
        f"{prefix}_left_fraction": float(mask[:, : evidence.shape[1] // 2].mean()),
        f"{prefix}_right_fraction": float(mask[:, evidence.shape[1] // 2 :].mean()),
    }


def _projection_matrix(input_dim: int, output_dim: int, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, output_dim)).astype(np.float32)
    return mat


def lesion_map_embedding(maps: dict[str, np.ndarray], dim: int = 128) -> np.ndarray:
    """Return a balanced 128-D X2 map embedding over all five lesion channels.

    Each lesion channel contributes four summary values plus a 5x5 pooled map.
    The 145-D balanced raw vector is projected deterministically to 128-D, so
    neovascularization remains represented without over-weighting a rare channel.
    """

    feats: list[float] = []
    for name in LESION_CHANNELS:
        arr = norm01(maps[name])
        feats.extend([float(arr.mean()), float(arr.std()), float(arr.max()), float((arr >= 0.35).mean())])
        feats.extend(cv2.resize(arr, (5, 5), interpolation=cv2.INTER_AREA).ravel().tolist())
    raw = np.asarray(feats, dtype=np.float32)
    raw = raw - float(raw.mean())
    raw = raw / max(float(np.linalg.norm(raw)), 1e-8)
    if raw.size == dim:
        return raw.astype(np.float32)
    projected = raw @ _projection_matrix(raw.size, dim)
    projected = projected - float(projected.mean())
    projected = projected / max(float(np.linalg.norm(projected)), 1e-8)
    return projected.astype(np.float32)


def image_embedding(rgb: np.ndarray, dim: int = 128) -> np.ndarray:
    feats: list[float] = []
    rgbf = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for arr, bins, scale in [
        (rgbf[..., 0], 12, (0, 1)),
        (rgbf[..., 1], 12, (0, 1)),
        (rgbf[..., 2], 12, (0, 1)),
        (hsv[..., 0], 12, (0, 180)),
        (hsv[..., 1], 12, (0, 255)),
        (hsv[..., 2], 12, (0, 255)),
    ]:
        hist, _ = np.histogram(arr, bins=bins, range=scale, density=False)
        hist = hist.astype(np.float32) / max(float(hist.sum()), 1e-8)
        feats.extend(hist.tolist())
    green = rgbf[..., 1]
    small = cv2.resize(green, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    feats.extend(cv2.dct(small)[:7, :7].ravel().tolist())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    for grid in [4, 6]:
        feats.extend(cv2.resize(edges, (grid, grid), interpolation=cv2.INTER_AREA).ravel().tolist())
    for channel in [lab[..., 0], lab[..., 1], lab[..., 2], edges]:
        channel = channel.astype(np.float32)
        feats.extend([
            float(channel.mean()),
            float(channel.std()),
            float(np.percentile(channel, 10)),
            float(np.percentile(channel, 90)),
        ])
    vec = np.asarray(feats, dtype=np.float32)
    if vec.size < dim:
        vec = np.pad(vec, (0, dim - vec.size))
    vec = vec[:dim]
    vec = vec - float(vec.mean())
    vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
    return vec.astype(np.float32)
