"""
tracer.py
Trace a black/white line-art PNG into an SVG made of:
  - ONE black path = the ink strokes (traced as-is)
  - many WHITE paths = each enclosed region between the ink lines

This reproduces the structure of the reference file `preview.svg`:
a single compound black path for the line art, plus one filled-white
path per colorable region, all sharing the same coordinate transform
so they overlay perfectly. Downstream tools can then target each
white <path> individually (e.g. to fill it with a color).

Requires the `potrace` CLI to be installed on PATH.
    Debian/Ubuntu:  apt-get install -y potrace
"""

from __future__ import annotations

import subprocess
import tempfile
import os
import re
import shutil
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage


class PotraceNotFoundError(RuntimeError):
    pass


@dataclasses.dataclass
class TraceConfig:
    # Output canvas size (matches the target pipeline: 600x800)
    width: int = 600
    height: int = 800

    # Pixel value below this is considered "ink" (0-255, grayscale)
    ink_threshold: int = 128

    # Regions (white areas) smaller than this many pixels are treated as
    # anti-aliasing dust: instead of being traced as their own path, every
    # pixel of the region is merged into whichever larger region is
    # spatially nearest (so it still gets colored later, just as part of
    # its neighbor rather than standing alone). Raise this if you see tiny
    # stray white slivers merged into the wrong neighbor.
    min_region_area: int = 12

    # potrace smoothing params (see `potrace --help`)
    turdsize: int = 2          # suppress speckles up to this many pixels
    alphamax: float = 1.0      # corner smoothing (0=polygon, 1.33=very smooth)
    opttolerance: float = 0.2  # curve optimization tolerance

    # 4-connectivity keeps regions that touch only diagonally separate
    # (safer: avoids "leaking" white paint across a 1px pinch in the ink).
    connectivity: int = 4

    # If True, resize the source PNG (keeping content, not cropping) to
    # exactly (width, height) before tracing. If False, the source must
    # already match (width, height) or a ValueError is raised.
    auto_resize: bool = True


@dataclasses.dataclass
class TraceResult:
    svg_path: str
    width: int
    height: int
    num_regions_total: int
    num_regions_traced: int
    num_regions_merged_as_noise: int


def _check_potrace() -> str:
    exe = shutil.which("potrace")
    if not exe:
        raise PotraceNotFoundError(
            "potrace CLI not found on PATH. Install it first, e.g.\n"
            "  apt-get update && apt-get install -y potrace"
        )
    return exe


def _write_pbm(mask: np.ndarray, path: str) -> None:
    """mask: 2D uint8 array, 1 = black/foreground pixel, 0 = background."""
    h, w = mask.shape
    packed = np.packbits(mask, axis=1)
    with open(path, "wb") as f:
        f.write(f"P4\n{w} {h}\n".encode("ascii"))
        f.write(packed.tobytes())


_PATH_RE = re.compile(r'<path\s+d="([^"]+)"', re.DOTALL)


def _potrace_svg_path(
    mask: np.ndarray,
    width: int,
    height: int,
    cfg: TraceConfig,
    workdir: str,
    tag: str,
) -> Optional[str]:
    """Run potrace on a single binary mask, return the SVG path 'd' string
    (or None if the mask traced to nothing, e.g. an all-zero mask)."""
    if not mask.any():
        return None

    pbm_path = os.path.join(workdir, f"{tag}.pbm")
    svg_path = os.path.join(workdir, f"{tag}.svg")
    _write_pbm(mask, pbm_path)

    cmd = [
        "potrace",
        "-s",  # svg backend
        "-o", svg_path,
        "-u", "1",                      # 1 unit = 1 pixel (no extra scaling)
        "-W", f"{width}pt",
        "-H", f"{height}pt",
        "-t", str(cfg.turdsize),
        "-a", str(cfg.alphamax),
        "-O", str(cfg.opttolerance),
        pbm_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Collect every path's d attribute (usually just one, but be safe)
    ds = _PATH_RE.findall(content)
    if not ds:
        return None
    return " ".join(d.strip() for d in ds)


def _load_ink_mask(png_path: str, cfg: TraceConfig) -> np.ndarray:
    im = Image.open(png_path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        # Composite onto white so transparent areas count as background,
        # not ink.
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    im = im.convert("L")

    if im.size != (cfg.width, cfg.height):
        if not cfg.auto_resize:
            raise ValueError(
                f"Image size {im.size} != target ({cfg.width}, {cfg.height}) "
                f"and auto_resize=False"
            )
        im = im.resize((cfg.width, cfg.height), Image.LANCZOS)

    arr = np.array(im)
    ink = (arr < cfg.ink_threshold).astype(np.uint8)
    return ink


def _label_regions(
    bg: np.ndarray, cfg: TraceConfig
) -> tuple[np.ndarray, list[int], int, int]:
    """Label connected white regions in `bg` (1 = background/colorable pixel,
    0 = ink). Any region smaller than `cfg.min_region_area` is merged into
    whichever larger region is spatially nearest to it -- every one of its
    pixels is reassigned to that neighbor's label -- rather than being
    dropped. If every region happens to be "small" (no larger neighbor
    exists to merge into), they are all left as their own separate regions
    so no white pixel is ever silently discarded.

    Returns (label_map, final_labels, total_regions_before_merge, num_merged).
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bg, connectivity=cfg.connectivity
    )
    # label 0 is the "background of bg" == the ink pixels; skip it.
    region_labels = list(range(1, n_labels))
    total_regions = len(region_labels)

    small_labels = [lbl for lbl in region_labels if stats[lbl, cv2.CC_STAT_AREA] < cfg.min_region_area]
    keep_labels = [lbl for lbl in region_labels if stats[lbl, cv2.CC_STAT_AREA] >= cfg.min_region_area]

    merged = labels.copy()
    num_merged = 0

    if small_labels and keep_labels:
        # For every pixel, find the nearest pixel belonging to a "keep"
        # (large-enough) region, then reassign each small region's pixels
        # to whichever keep-region got the most votes among its own pixels
        # -- so the whole sliver moves as one piece into one neighbor.
        seed_mask = np.isin(labels, keep_labels)
        _, nearest_idx = ndimage.distance_transform_edt(~seed_mask, return_indices=True)
        nearest_label = labels[nearest_idx[0], nearest_idx[1]]
        for lbl in small_labels:
            pixels = labels == lbl
            votes = nearest_label[pixels]
            target = int(np.bincount(votes).argmax())
            merged[pixels] = target
            num_merged += 1

    final_labels = sorted(int(l) for l in np.unique(merged) if l != 0)
    return merged, final_labels, total_regions, num_merged


def trace_lineart(
    png_path: str,
    out_svg_path: str,
    cfg: Optional[TraceConfig] = None,
) -> TraceResult:
    """Main entry point: trace one PNG line-art file into a region-separated SVG."""
    cfg = cfg or TraceConfig()
    _check_potrace()

    ink = _load_ink_mask(png_path, cfg)
    H, W = ink.shape
    bg = (1 - ink).astype(np.uint8)

    with tempfile.TemporaryDirectory(prefix="lineart_trace_") as workdir:
        # 1) The black line-art layer: trace the ink mask as ONE path.
        black_d = _potrace_svg_path(ink, W, H, cfg, workdir, "ink")

        # 2) The white regions: label connected background components
        #    (merging any too-small ones into their nearest neighbor), then
        #    trace each surviving region separately so it can later be
        #    targeted/colored on its own.
        merged_labels, final_labels, total_regions, num_merged = _label_regions(bg, cfg)

        white_ds = []
        for lbl in final_labels:
            region_mask = (merged_labels == lbl).astype(np.uint8)
            d = _potrace_svg_path(region_mask, W, H, cfg, workdir, f"region_{lbl}")
            if d:
                white_ds.append(d)

    svg = _assemble_svg(W, H, black_d, white_ds)
    Path(out_svg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return TraceResult(
        svg_path=out_svg_path,
        width=W,
        height=H,
        num_regions_total=total_regions,
        num_regions_traced=len(white_ds),
        num_regions_merged_as_noise=num_merged,
    )


def _assemble_svg(width: int, height: int, black_d: Optional[str], white_ds: list[str]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'fill="none" xmlns="http://www.w3.org/2000/svg">\n',
        # potrace's own transform for -u 1 -W {w}pt -H {h}pt is
        # translate(0,H) scale(1,-1) -- flips its internal (origin-bottom-left)
        # coordinates back to normal SVG (origin-top-left) space.
        f'<g transform="translate(0,{height}) scale(1,-1)">\n',
    ]
    if black_d:
        parts.append(f'<path d="{black_d}" fill="black"/>\n')
    for d in white_ds:
        parts.append(f'<path d="{d}" fill="white"/>\n')
    parts.append("</g>\n</svg>\n")
    return "".join(parts)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Trace a line-art PNG into a region-separated SVG")
    ap.add_argument("input", help="input PNG file")
    ap.add_argument("output", help="output SVG file")
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--ink-threshold", type=int, default=128)
    ap.add_argument("--min-region-area", type=int, default=12)
    args = ap.parse_args()

    cfg = TraceConfig(
        width=args.width,
        height=args.height,
        ink_threshold=args.ink_threshold,
        min_region_area=args.min_region_area,
    )
    result = trace_lineart(args.input, args.output, cfg)
    print(result)
