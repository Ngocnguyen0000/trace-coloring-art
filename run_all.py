"""
run_all.py
Bam nut Run la chay: quet toan bo file .png trong input/ (ke ca cac thu muc
con long nhau), trace tung anh, va ghi ra output/ theo dung cau truc thu
muc tuong ung. Khong can truyen tham so dong lenh.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import time
import traceback
from pathlib import Path
from dataclasses import asdict

from tracer import TraceConfig, trace_lineart, _check_potrace

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
WORKERS = 2


def _process_one(png_path: str, out_svg_path: str, cfg: TraceConfig) -> dict:
    t0 = time.time()
    try:
        result = trace_lineart(png_path, out_svg_path, cfg)
        return {
            "file": png_path,
            "ok": True,
            "output": out_svg_path,
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


def main() -> None:
    cfg = TraceConfig()
    _check_potrace()

    pngs = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() == ".png")
    if not pngs:
        print(f"Khong tim thay file .png nao trong {INPUT_DIR}")
        return

    jobs = []
    for p in pngs:
        rel = p.relative_to(INPUT_DIR)
        out_svg = OUTPUT_DIR / rel.with_suffix(".svg")
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(p), str(out_svg)))

    print(f"Tim thay {len(jobs)} anh. Dang xu ly voi {WORKERS} workers...")

    results = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(_process_one, png, svg, cfg): png for png, svg in jobs
        }
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            status = "OK " if r["ok"] else "LOI"
            print(f"[{status}] {r['file']} ({r['seconds']}s)")

    results.sort(key=lambda r: r["file"])
    ok = sum(1 for r in results if r["ok"])
    summary = {
        "total": len(results),
        "ok": ok,
        "failed": len(results) - ok,
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nXong: {ok}/{len(results)} thanh cong, {len(results) - ok} loi.")
    print(f"Xem chi tiet trong {OUTPUT_DIR / '_report.json'}")


if __name__ == "__main__":
    main()
