# lineart_trace

Trace a black/white line-art PNG (kawaii / flower coloring-page style) into
an SVG shaped exactly like `preview.svg`:

- **one black `<path>`** = the ink line art, traced as-is
- **many white `<path>`s** = each enclosed region between the lines,
  traced separately so a later step can select/fill each one individually

All paths share the same `<g transform="translate(0,H) scale(1,-1)">`,
so they overlay pixel-perfectly.

## How it works

1. Threshold the PNG to a binary "ink" mask (dark pixels = ink).
2. Run `potrace` on the ink mask → the single black path.
3. Invert the mask → background pixels. Label connected components
   (`cv2.connectedComponentsWithStats`, 4-connectivity) → each component
   is one colorable region.
4. Run `potrace` on each region's mask individually (same canvas size,
   so the coordinate transform is identical every time) → one white path
   per region. Regions smaller than `min_region_area` px are treated as
   anti-aliasing dust: every pixel of the tiny region is merged into
   whichever larger region is spatially nearest, so it still ends up
   traced/colored as part of a neighbor instead of being dropped.
5. Assemble the final SVG: black path first, white paths on top.

Tested end-to-end against `preview.svg`'s companion PNG
(`lineart__10_.png`, 600×800) — traced output visually matches the
original line art closely, curves stay smooth (potrace's own bezier
fitting, same as the reference file uses).

## Install

```bash
apt-get update && apt-get install -y potrace   # system dependency
pip install -r requirements.txt
```

> **Render.com note:** Render's native Python runtime does NOT run
> `apt-get` for you. Either switch that service to a **Docker** deploy
> (add `RUN apt-get update && apt-get install -y potrace` to your
> Dockerfile) or use a Render **build script** that vendors a static
> potrace binary. Docker is the simpler path.

## CLI usage

Single file:
```bash
python3 tracer.py input.png output.svg
python3 tracer.py input.png output.svg --ink-threshold 140 --min-region-area 20
```

Batch (folder of PNGs -> folder of SVGs + `_report.json`):
```bash
python3 batch.py ./pngs_in ./svgs_out --workers 4
```

## Flask integration (step 1 upload + step 3 batch)

```python
from flask import Flask
from flask_blueprint import trace_bp

app = Flask(__name__)
app.register_blueprint(trace_bp, url_prefix="/trace")
```

Routes added:
- `GET  /trace/`        simple HTML upload form (for manual testing)
- `POST /trace/single`  form field `file` = one .png -> returns one .svg
- `POST /trace/batch`   form field `file` = one .zip of .png -> returns .zip of .svg

Both accept optional form fields to override defaults:
`width`, `height`, `ink_threshold`, `min_region_area`, and (`batch` only) `workers`.

To fold this into your existing pipeline as "step 2" (between PNG upload
and the color-fill step), call `trace_lineart()` directly instead of going
through HTTP — it's just a plain function:

```python
from tracer import trace_lineart, TraceConfig
result = trace_lineart("uploads/foo.png", "traced/foo.svg", TraceConfig())
print(result.num_regions_traced, "regions ready to color")
```

## Tuning knobs (`TraceConfig`)

| field | default | effect |
|---|---|---|
| `ink_threshold` | 128 | lower = only very dark pixels count as ink (thinner lines) |
| `min_region_area` | 12 | raise if tiny stray white specks are merging into the wrong neighbor as their own visible blob; lower if small real regions (e.g. eye highlights) are getting merged away when they should stay separate |
| `turdsize` | 2 | potrace speckle suppression on the traced curves themselves |
| `alphamax` | 1.0 | corner smoothing; 0 = sharp polygon corners, 1.33 = very rounded |
| `opttolerance` | 0.2 | bezier curve fit tolerance; higher = fewer/smoother curve segments |
| `connectivity` | 4 | 4 keeps diagonally-touching regions separate (safer); 8 merges them |

## Files

- `tracer.py` — core engine + single-file CLI
- `batch.py` — batch folder processing (multiprocess) + CLI
- `flask_blueprint.py` — drop-in Flask routes for upload + batch
- `requirements.txt` — Python deps (potrace itself is a separate system package)
