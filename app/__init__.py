import functools
import hashlib
import logging
import os

from flask import Flask, jsonify, render_template, request

MAX_UPLOAD_MB = 12


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 60 * 60 * 24 * 7
    app.json.sort_keys = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from app.routes import main
    app.register_blueprint(main)

    # Cache-bust static assets: url_for('static', ...) gets ?v=<content hash>,
    # so the week-long browser cache never serves a stale file after a deploy.
    @functools.lru_cache(maxsize=None)
    def static_hash(filename):
        try:
            with open(os.path.join(app.static_folder, filename), "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:10]
        except OSError:
            return None

    @app.url_defaults
    def version_static_urls(endpoint, values):
        if endpoint == "static" and "filename" in values and "v" not in values:
            version = static_hash(values["filename"])
            if version:
                values["v"] = version

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"status": "error", "message": f"Image is too large (max {MAX_UPLOAD_MB} MB)."}), 413

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": "Not found"}), 404
        return render_template("404.html"), 404

    # Load the ONNX model at startup so the first visitor doesn't wait for it.
    if os.environ.get("WARM_MODEL", "1") == "1":
        from src.features.breed_predictor import predictor
        try:
            predictor.session
            app.logger.info("Breed model loaded")
        except Exception:
            app.logger.exception("Could not load breed model; /api/analyze will be unavailable")

    return app


app = create_app()
