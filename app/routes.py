import logging

from flask import Blueprint, jsonify, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError

from src.features.breed_predictor import PROFILES, PROFILES_BY_NAME, estimate_body, measurements_for_weight, predictor
from src.features.gear_recommender import ProfileError, parse_profile, recommend

Image.MAX_IMAGE_PIXELS = 40_000_000  # reject decompression bombs

main = Blueprint("main", __name__)
log = logging.getLogger(__name__)


def error(message, status=400):
    return jsonify({"status": "error", "message": message}), status


FAQ = [
    ("How accurate is the breed recognition?",
     "On everyday photos it names the exact breed about four times in five, and the right breed is in its top three 97% of the time. "
     "If it picks wrong, choose the right breed from the list and everything updates."),
    ("Does it measure my dog from the photo?",
     "Not directly. A single photo has no sense of scale, so a close-up Chihuahua and a distant Great Dane can look the same size. "
     "We recognize the breed and build, then estimate measurements from typical adult sizes. "
     "If you have a tape measure, enter the real numbers and every size recalculates."),
    ("My dog is a mix. Will it still work?",
     "Yes. When the photo looks like more than one breed, we blend the closest matches to estimate size. You can also type in the breed or measurements yourself."),
    ("Do you sell the gear?",
     "No. Each pick links to Amazon and Chewy so you can check the current price and buy wherever you like."),
    ("What happens to my photo?",
     "It is analyzed in memory and discarded as soon as the result comes back. We never store it."),
    ("My puppy is still growing. What should I get?",
     "Tell us in the questions and we favor gear with more adjustment. Re-measure every two or three weeks and size up when straps run out of room."),
    ("Why did it pick a harness over a collar?",
     "Harnesses spread pressure across the chest instead of the throat. For pullers, small dogs and flat-faced breeds we always attach the leash to the harness and treat the collar as an ID tag holder."),
    ("Are the prices exact?",
     "No, they are typical street prices and change often. Use the retailer links to see what it costs today."),
]


# -- PAGE ROUTES --

@main.route("/")
def index():
    return render_template("index.html", faq=FAQ)


@main.route("/demo")
def demo():
    return render_template("demo.html", breeds=sorted(p["name"] for p in PROFILES.values()))


@main.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")


@main.route("/company")
def company():
    return render_template("company.html")


@main.route("/faq")
def faq():
    return render_template("faq.html", faq=FAQ)


# Old URLs from the previous site.
@main.route("/about")
@main.route("/team")
@main.route("/why-us")
def old_company_pages():
    return redirect(url_for("main.company"), 301)


@main.route("/features")
def old_features_page():
    return redirect(url_for("main.index"), 301)


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
