"""
app.py
Flask entrypoint for deployment (Docker / Render / gunicorn): registers the
tracer blueprint at the site root, so `GET /` serves the upload form
directly and `POST /single` / `POST /batch` work with no extra prefix.

Local dev:   python3 app.py            (http://localhost:8000)
Production:  gunicorn app:app
"""

from flask import Flask

from flask_blueprint import trace_bp

app = Flask(__name__)
app.register_blueprint(trace_bp, url_prefix="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
