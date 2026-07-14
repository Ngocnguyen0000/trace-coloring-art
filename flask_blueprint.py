"""
flask_blueprint.py
Drop-in Flask blueprint exposing the line-art tracer as a web step:

  GET  /trace/                 -> simple upload form (for standalone testing)
  POST /trace/single           -> upload ONE PNG, get back ONE SVG
  POST /trace/batch            -> upload a .zip of PNGs, get back a .zip of SVGs

Wire it into an existing app with:

    from flask_blueprint import trace_bp
    app.register_blueprint(trace_bp, url_prefix="/trace")

Requires: Flask, and everything tracer.py needs (see requirements.txt),
plus the `potrace` binary on PATH (apt-get install -y potrace).
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from flask import Blueprint, request, send_file, jsonify, Response

from tracer import TraceConfig, trace_lineart, PotraceNotFoundError
from batch import batch_trace

trace_bp = Blueprint("trace_bp", __name__)

ALLOWED_EXT = {".png"}


def _cfg_from_request() -> TraceConfig:
    """Let callers override defaults via form fields, e.g. ink_threshold=140."""
    def _f(name, default, cast):
        val = request.form.get(name)
        return cast(val) if val not in (None, "") else default

    return TraceConfig(
        width=_f("width", 600, int),
        height=_f("height", 800, int),
        ink_threshold=_f("ink_threshold", 128, int),
        min_region_area=_f("min_region_area", 12, int),
    )


@trace_bp.get("/")
def upload_form() -> Response:
    return Response(
        """
        <h3>Trace single PNG</h3>
        <form method="post" action="single" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/png" required>
          <button type="submit">Trace</button>
        </form>
        <h3>Batch trace (zip of PNGs)</h3>
        <form method="post" action="batch" enctype="multipart/form-data">
          <input type="file" name="file" accept=".zip" required>
          <button type="submit">Trace batch</button>
        </form>
        """,
        mimetype="text/html",
    )


@trace_bp.post("/single")
def trace_single():
    f = request.files.get("file")
    if not f or Path(f.filename).suffix.lower() not in ALLOWED_EXT:
        return jsonify({"error": "upload one .png file as 'file'"}), 400

    cfg = _cfg_from_request()
    with tempfile.TemporaryDirectory(prefix="trace_single_") as tmp:
        in_path = os.path.join(tmp, "input.png")
        out_path = os.path.join(tmp, "output.svg")
        f.save(in_path)
        try:
            trace_lineart(in_path, out_path, cfg)
        except PotraceNotFoundError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

        buf = io.BytesIO(Path(out_path).read_bytes())
    buf.seek(0)
    out_name = Path(f.filename).stem + ".svg"
    return send_file(buf, mimetype="image/svg+xml", as_attachment=True, download_name=out_name)


@trace_bp.post("/batch")
def trace_batch():
    f = request.files.get("file")
    if not f or Path(f.filename).suffix.lower() != ".zip":
        return jsonify({"error": "upload a .zip of .png files as 'file'"}), 400

    cfg = _cfg_from_request()
    workers = int(request.form.get("workers", 4))

    with tempfile.TemporaryDirectory(prefix="trace_batch_") as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir, exist_ok=True)

        zip_path = os.path.join(tmp, "upload.zip")
        f.save(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                # flatten any folder structure, skip non-PNGs, guard traversal
                name = os.path.basename(member)
                if not name.lower().endswith(".png"):
                    continue
                target = os.path.join(in_dir, name)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        try:
            summary = batch_trace(in_dir, out_dir, cfg, workers=workers)
        except PotraceNotFoundError as e:
            return jsonify({"error": str(e)}), 500

        out_zip_path = os.path.join(tmp, "traced.zip")
        with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in Path(out_dir).glob("*.svg"):
                zf.write(p, arcname=p.name)
            zf.write(Path(out_dir) / "_report.json", arcname="_report.json")

        buf = io.BytesIO(Path(out_zip_path).read_bytes())
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="traced.zip")
