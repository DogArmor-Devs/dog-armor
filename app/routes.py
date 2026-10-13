import logging

from flask import Blueprint, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.features.breed_predictor import PROFILES, PROFILES_BY_NAME, estimate_body, measurements_for_weight, predictor
from src.features.gear_recommender import ProfileError, parse_profile, recommend

Image.MAX_IMAGE_PIXELS = 40_000_000  # reject decompression bombs

main = Blueprint("main", __name__)
log = logging.getLogger(__name__)


def error(message, status=400):
    return jsonify({"status": "error", "message": message}), status


# -- PAGE ROUTES --

@main.route("/")
def index():
    return render_template("index.html")


@main.route("/demo")
def demo():
    return render_template("demo.html", breeds=sorted(p["name"] for p in PROFILES.values()))


@main.route("/features")
def features():
    return render_template("features.html")


@main.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")


@main.route("/why-us")
def why_us():
    return render_template("why-us.html")


@main.route("/about")
def about():
    return render_template("about.html")


@main.route("/team")
def team():
    return render_template("team.html")


@main.route("/healthz")
def healthz():
    return {"status": "ok"}


# -- API --

@main.route("/api/analyze", methods=["POST"])
def analyze():
    """Photo -> breed guesses, dog-presence check and body measurement estimate."""
    file = request.files.get("image")
    if not file or not file.filename:
        return error("Please choose a photo of your dog.")
    try:
        image = Image.open(file.stream)
        image.draft("RGB", (768, 768))  # fast JPEG downscale on decode
        result = predictor.analyze(image)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return error("That file doesn't look like a supported image (try JPG, PNG or WebP).")
    except Exception:
        log.exception("Breed analysis failed")
        return error("We couldn't analyze that photo. Please try another one.", 500)

    log.info("analyze: top=%s dog_score=%.2f", result["breeds"][0]["name"], result["dog_score"])
    return jsonify({"status": "success", **result})


@main.route("/api/estimate")
def estimate():
    """Body estimate for a chosen breed, or for a known weight (+ optional breed build)."""
    profile = PROFILES_BY_NAME.get(request.args.get("breed", "").strip().lower())
    weight = request.args.get("weight", type=float)
    if weight is not None:
        if not 1 <= weight <= 250:
            return error("Weight must be between 1 and 250 lbs.")
        build = profile["build"] if profile else "standard"
        result = {"weight_lbs": weight, "build": build, "measurements": measurements_for_weight(weight, build)}
    elif profile:
        result = estimate_body([(profile, 1.0)])
    else:
        return error("Unknown breed. Pick one from the list or enter your dog's weight.")
    traits = {k: profile[k] for k in ("brachycephalic", "coat", "build")} if profile else None
    return jsonify({"status": "success", "estimate": result, "traits": traits})


@main.route("/api/recommend", methods=["POST"])
def recommend_gear():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Expected a JSON body.")

    # Breed traits come from our profile table, not the client, when we know the breed.
    known = PROFILES_BY_NAME.get(str(data.get("breed", "")).strip().lower())
    if known:
        data = {**data, "brachycephalic": known["brachycephalic"], "build": known["build"]}
    try:
        profile = parse_profile(data)
    except ProfileError as exc:
        return error(str(exc))

    result = recommend(profile)
    log.info("recommend: breed=%s budget=%s total=%s", profile["breed"], profile["budget"], result["total_price"])
    return jsonify({"status": "success", **result})


@main.route("/api/breeds")
def breeds():
    return jsonify(sorted(p["name"] for p in PROFILES.values()))
