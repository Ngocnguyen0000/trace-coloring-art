"""
batch.py
Batch-trace a whole folder of line-art PNGs into region-separated SVGs.

Usage:
    python3 batch.py INPUT_DIR OUTPUT_DIR [--workers 4] [--width 600] [--height 800]

Writes one .svg per .png (same basename) into OUTPUT_DIR, plus a
`_report.json` summarizing successes/failures/region counts.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import traceback
from pathlib import Path
from dataclasses import asdict

from tracer import TraceConfig, trace_lineart, PotraceNotFoundError, _check_potrace

IMAGE_EXTS = {".png"}


def _process_one(png_path: str, out_dir: str, cfg: TraceConfig) -> dict:
    name = Path(png_path).stem
    out_svg = str(Path(out_dir) / f"{name}.svg")
    t0 = time.time()
    try:
        result = trace_lineart(png_path, out_svg, cfg)
        return {
            "file": png_path,
            "ok": True,
            "output": out_svg,
            "seconds": round(time.time() - t0, 2),
            **{k: v for k, v in asdict(result).items() if k != "svg_path"},
        }
    except Exception as e:
        return {
            "file": png_path,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.time() - t0, 2),
        }


def batch_trace(
    input_dir: str,
    output_dir: str,
    cfg: TraceConfig | None = None,
    workers: int = 4,
) -> dict:
    cfg = cfg or TraceConfig()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(
        p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not pngs:
        return {"total": 0, "ok": 0, "failed": 0, "results": []}

    results = []
    # Fail fast with a clear message if potrace isn't installed, instead of
    # spawning N workers that will all fail the same way.
    _check_potrace()

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_process_one, str(p), str(out_dir), cfg): p for p in pngs
        }
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["file"])
    ok = sum(1 for r in results if r["ok"])
    summary = {
        "total": len(results),
        "ok": ok,
        "failed": len(results) - ok,
        "results": results,
    }
    with open(out_dir / "_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch-trace a folder of line-art PNGs")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--workers", type=int, default=4)
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
    summary = batch_trace(args.input_dir, args.output_dir, cfg, workers=args.workers)
    print(f"Done: {summary['ok']}/{summary['total']} ok, "
          f"{summary['failed']} failed. See _report.json for details.")
