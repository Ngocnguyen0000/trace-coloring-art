"""
colorizer.py
Trace a line-art PNG AND auto-fill each region with color sampled from a
reference color image -- all in one step.

This is the Python port of the browser tool "SVG Color Transfer"
(https://github.com/Ngocnguyen0000/SVG-Color-Transfer), fused directly
into the tracer pipeline: because we still hold each region's pixel mask
at trace time, we sample the reference image through that exact mask --
no SVG rasterization / segmentation-color roundtrip needed, so region ->
color mapping is pixel-perfect.

Feature parity with the JS tool:
  - sample mode: "mean" (average color in region) or "dominant"
    (biggest 4-bit color bucket, resistant to line-art noise bleeding in)
  - saturation / brightness boost (HSL space)
  - palette limiting to 1..200 colors (weighted median-cut, weight =
    region pixel area)
  - dark outline is kept as-is (the black ink path stays black)

Usage (single file):
    python colorizer.py lineart.png color_ref.png output.svg
    python colorizer.py lineart.png color_ref.png output.svg \
        --sample-mode dominant --max-colors 24 --saturation 10 --brightness 4

Usage (batch: all three args are folders, files matched by basename):
    python colorizer.py ./lineart_dir ./color_dir ./out_dir --workers 4
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image

from tracer import (
    TraceConfig,
    _check_potrace,
    _load_ink_mask,
    _potrace_svg_path,
    _label_regions,
)


@dataclasses.dataclass
class ColorConfig:
    # "mean"  = average of all reference pixels inside the region
    # "dominant" = most common 4-bit color bucket (ignores stray line pixels)
    sample_mode: str = "mean"

    # Limit the final SVG to at most this many distinct colors (1..200).
    # 200 effectively means "no limit" for typical coloring pages.
    max_colors: int = 200

    # Percentage boosts applied in HSL space, same as the JS tool sliders.
    saturation: float = 10.0   # +10% saturation
    brightness: float = 4.0    # +4% lightness

    # Erode each region mask by this many pixels before sampling, so
    # anti-aliased line pixels at the region border don't darken the color.
    # If erosion empties a small region, the un-eroded mask is used.
    erode_px: int = 1

    # Fill used when a region has no usable reference pixels.
    fallback_color: tuple[int, int, int] = (245, 245, 245)


@dataclasses.dataclass
class ColorizeResult:
    svg_path: str
    width: int
    height: int
    num_regions_total: int
    num_regions_traced: int
    num_regions_merged_as_noise: int
    num_regions_colored: int
    palette_count: int


# ---------------------------------------------------------------------------
# color helpers (ports of the JS tool's rgbToHsl / hslToRgb / adjustColor)
# ---------------------------------------------------------------------------

def _rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return h / 6.0, s, l


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    if s == 0:
        return l * 255.0, l * 255.0, l * 255.0

    def hue2rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (
        hue2rgb(p, q, h + 1 / 3) * 255.0,
        hue2rgb(p, q, h) * 255.0,
        hue2rgb(p, q, h - 1 / 3) * 255.0,
    )


def _adjust_color(rgb: tuple[float, float, float], cfg: ColorConfig) -> tuple[float, float, float]:
    h, s, l = _rgb_to_hsl(*rgb)
    s = min(1.0, max(0.0, s * (1.0 + cfg.saturation / 100.0)))
    l = min(1.0, max(0.0, l + cfg.brightness / 100.0))
    return _hsl_to_rgb(h, s, l)


def _hex(rgb) -> str:
    return "#" + "".join(f"{min(255, max(0, round(v))):02X}" for v in rgb)


# ---------------------------------------------------------------------------
# palette quantization (port of the JS tool's weighted median-cut)
# ---------------------------------------------------------------------------

def _weighted_average(items: list[tuple[tuple[float, float, float], float]]) -> tuple[float, float, float]:
    total = sum(w for _, w in items)
    if not total:
        return (245.0, 245.0, 245.0)
    r = sum(c[0] * w for c, w in items) / total
    g = sum(c[1] * w for c, w in items) / total
    b = sum(c[2] * w for c, w in items) / total
    return (r, g, b)


def _channel_range(items, ch: int) -> float:
    vals = [c[ch] for c, _ in items]
    return max(vals) - min(vals)


def _median_cut_palette(colors, weights, max_colors: int):
    import math

    limit = max(1, min(200, round(max_colors)))
    items = [(tuple(c), max(1.0, float(w))) for c, w in zip(colors, weights)]
    if not items:
        return [(245.0, 245.0, 245.0)]
    if limit <= 1:
        return [_weighted_average(items)]

    boxes = [items]
    while len(boxes) < limit:
        best_idx, best_score = -1, -1.0
        for i, box in enumerate(boxes):
            if len(box) < 2:
                continue
            rng = max(_channel_range(box, 0), _channel_range(box, 1), _channel_range(box, 2))
            total_w = sum(w for _, w in box)
            score = rng * math.log2(total_w + 1)
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx == -1:
            break

        box = boxes.pop(best_idx)
        ranges = [_channel_range(box, ch) for ch in range(3)]
        ch = ranges.index(max(ranges))
        box.sort(key=lambda item: item[0][ch])
        total_w = sum(w for _, w in box)
        acc, split = 0.0, 1
        for i in range(len(box) - 1):
            acc += box[i][1]
            if acc >= total_w / 2:
                split = i + 1
                break
        left, right = box[:split], box[split:]
        if not left or not right:
            boxes.append(box)
            break
        boxes.append(left)
        boxes.append(right)

    return [_weighted_average(b) for b in boxes]


def _quantize(colors, weights, max_colors: int):
    """Map each color to its nearest palette entry. Returns (mapped, palette_count)."""
    palette = _median_cut_palette(colors, weights, max_colors)
    mapped = []
    for c in colors:
        best, best_d = palette[0], float("inf")
        for p in palette:
            d = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2
            if d < best_d:
                best_d, best = d, p
        mapped.append(best)
    palette_count = len({_hex(c) for c in mapped})
    return mapped, palette_count


# ---------------------------------------------------------------------------
# reference image sampling
# ---------------------------------------------------------------------------

def _load_reference(color_png: str, width: int, height: int):
    """Return (rgb array HxWx3 float32, alpha array HxW uint8) resized to canvas."""
    im = Image.open(color_png)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB").convert("RGBA")
    if im.size != (width, height):
        im = im.resize((width, height), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    return arr[:, :, :3].astype(np.float32), arr[:, :, 3]


def _sample_region_color(
    region_mask: np.ndarray,
    ref_rgb: np.ndarray,
    ref_alpha: np.ndarray,
    cfg: ColorConfig,
) -> Optional[tuple[float, float, float]]:
    """Sample the reference color for one region. Returns None if no usable pixels."""
    mask = region_mask
    if cfg.erode_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(region_mask, kernel, iterations=cfg.erode_px)
        if eroded.any():
            mask = eroded

    sel = (mask == 1) & (ref_alpha >= 128)
    if not sel.any():
        # fall back to the raw mask before giving up
        sel = (region_mask == 1) & (ref_alpha >= 128)
        if not sel.any():
            return None

    pixels = ref_rgb[sel]  # N x 3

    if cfg.sample_mode == "dominant":
        # 4-bit bucket per channel, pick the biggest bucket, average inside it
        q = (pixels.astype(np.uint16) // 16)
        keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        best = int(np.argmax(counts))
        bucket_pixels = pixels[inverse == best]
        mean = bucket_pixels.mean(axis=0)
    else:
        mean = pixels.mean(axis=0)

    return (float(mean[0]), float(mean[1]), float(mean[2]))


# ---------------------------------------------------------------------------
# main pipeline: trace + colorize in one pass
# ---------------------------------------------------------------------------

def trace_and_colorize(
    lineart_png: str,
    color_png: str,
    out_svg_path: str,
    trace_cfg: Optional[TraceConfig] = None,
    color_cfg: Optional[ColorConfig] = None,
) -> ColorizeResult:
    """Trace `lineart_png` and fill each region with color sampled from
    `color_png` at the same position. Writes the colored SVG to `out_svg_path`."""
    trace_cfg = trace_cfg or TraceConfig()
    color_cfg = color_cfg or ColorConfig()
    _check_potrace()

    ink = _load_ink_mask(lineart_png, trace_cfg)
    H, W = ink.shape
    bg = (1 - ink).astype(np.uint8)

    ref_rgb, ref_alpha = _load_reference(color_png, W, H)

    with tempfile.TemporaryDirectory(prefix="lineart_color_") as workdir:
        black_d = _potrace_svg_path(ink, W, H, trace_cfg, workdir, "ink")

        merged_labels, final_labels, total_regions, num_merged = _label_regions(bg, trace_cfg)

        region_ds: list[str] = []
        region_colors: list[tuple[float, float, float]] = []
        region_weights: list[float] = []
        colored = 0

        for lbl in final_labels:
            region_mask = (merged_labels == lbl).astype(np.uint8)
            d = _potrace_svg_path(region_mask, W, H, trace_cfg, workdir, f"region_{lbl}")
            if not d:
                continue

            rgb = _sample_region_color(region_mask, ref_rgb, ref_alpha, color_cfg)
            if rgb is not None:
                rgb = _adjust_color(rgb, color_cfg)
                colored += 1
            else:
                rgb = tuple(float(v) for v in color_cfg.fallback_color)

            region_ds.append(d)
            region_colors.append(rgb)
            region_weights.append(float(region_mask.sum()))

    mapped, palette_count = _quantize(region_colors, region_weights, color_cfg.max_colors)
    fills = [_hex(c) for c in mapped]

    svg = _assemble_colored_svg(W, H, black_d, region_ds, fills)
    Path(out_svg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return ColorizeResult(
        svg_path=out_svg_path,
        width=W,
        height=H,
        num_regions_total=total_regions,
        num_regions_traced=len(region_ds),
        num_regions_merged_as_noise=num_merged,
        num_regions_colored=colored,
        palette_count=palette_count,
    )


def _assemble_colored_svg(
    width: int,
    height: int,
    black_d: Optional[str],
    region_ds: list[str],
    fills: list[str],
) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'fill="none" xmlns="http://www.w3.org/2000/svg">\n',
        f'<g transform="translate(0,{height}) scale(1,-1)">\n',
    ]
    if black_d:
        parts.append(f'<path d="{black_d}" fill="black"/>\n')
    for d, fill in zip(region_ds, fills):
        parts.append(f'<path d="{d}" fill="{fill}"/>\n')
    parts.append("</g>\n</svg>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# batch mode
# ---------------------------------------------------------------------------

def batch_colorize(
    lineart_dir: str,
    color_dir: str,
    out_dir: str,
    trace_cfg: Optional[TraceConfig] = None,
    color_cfg: Optional[ColorConfig] = None,
    workers: int = 4,
) -> dict:
    """Match PNGs in `lineart_dir` and `color_dir` by basename, colorize each
    pair, write SVGs (and _report.json) into `out_dir`."""
    import concurrent.futures as cf
    import json
    import time
    import traceback

    trace_cfg = trace_cfg or TraceConfig()
    color_cfg = color_cfg or ColorConfig()
    _check_potrace()

    line_dir, col_dir, o_dir = Path(lineart_dir), Path(color_dir), Path(out_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    color_by_stem = {
        p.stem: p for p in sorted(col_dir.iterdir()) if p.suffix.lower() in exts
    }

    pairs = []
    missing = []
    for p in sorted(line_dir.iterdir()):
        if p.suffix.lower() != ".png":
            continue
        ref = color_by_stem.get(p.stem)
        if ref is None:
            missing.append(p.name)
        else:
            pairs.append((p, ref))

    def _one(line_png: Path, ref_png: Path) -> dict:
        out_svg = str(o_dir / f"{line_png.stem}.svg")
        t0 = time.time()
        try:
            r = trace_and_colorize(str(line_png), str(ref_png), out_svg, trace_cfg, color_cfg)
            return {
                "file": str(line_png), "ref": str(ref_png), "ok": True,
                "output": out_svg, "seconds": round(time.time() - t0, 2),
                **{k: v for k, v in dataclasses.asdict(r).items() if k != "svg_path"},
            }
        except Exception as e:
            return {
                "file": str(line_png), "ref": str(ref_png), "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - t0, 2),
            }

    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, lp, rp) for lp, rp in pairs]
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["file"])
    ok = sum(1 for r in results if r["ok"])
    summary = {
        "total": len(results), "ok": ok, "failed": len(results) - ok,
        "missing_color_ref": missing, "results": results,
    }
    with open(o_dir / "_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Trace line-art PNG + auto-fill regions from a reference color image"
    )
    ap.add_argument("lineart", help="line-art PNG file (or folder in batch mode)")
    ap.add_argument("color", help="reference color image (or folder in batch mode)")
    ap.add_argument("output", help="output SVG file (or folder in batch mode)")
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--ink-threshold", type=int, default=128)
    ap.add_argument("--min-region-area", type=int, default=12)
    ap.add_argument("--sample-mode", choices=["mean", "dominant"], default="mean")
    ap.add_argument("--max-colors", type=int, default=200)
    ap.add_argument("--saturation", type=float, default=10.0, help="saturation boost %%")
    ap.add_argument("--brightness", type=float, default=4.0, help="brightness boost %%")
    ap.add_argument("--erode-px", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="batch mode only")
    args = ap.parse_args()

    trace_cfg = TraceConfig(
        width=args.width, height=args.height,
        ink_threshold=args.ink_threshold, min_region_area=args.min_region_area,
    )
    color_cfg = ColorConfig(
        sample_mode=args.sample_mode, max_colors=args.max_colors,
        saturation=args.saturation, brightness=args.brightness,
        erode_px=args.erode_px,
    )

    if Path(args.lineart).is_dir():
        summary = batch_colorize(
            args.lineart, args.color, args.output, trace_cfg, color_cfg, args.workers
        )
        print(f"Done: {summary['ok']}/{summary['total']} ok, {summary['failed']} failed.")
        if summary["missing_color_ref"]:
            print(f"No color ref found for: {', '.join(summary['missing_color_ref'])}")
    else:
        result = trace_and_colorize(args.lineart, args.color, args.output, trace_cfg, color_cfg)
        print(result)
