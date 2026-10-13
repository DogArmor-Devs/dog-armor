"""Breed recognition and body-measurement estimates from a dog photo.

Inference runs on onnxruntime with models/breed_classifier.onnx (see
export_onnx.py). Measurements are estimated from the predicted breed(s): each
breed has a typical adult weight range and build in
data/processed_data/breed_profiles.csv, and girths/lengths scale with the cube
root of body weight, adjusted for build (e.g. sighthounds have deep chests and
tucked bellies, bulldog types have thick necks).
"""

import csv
import logging
import math
import os
import threading
import time

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "breed_classifier.onnx")
PROFILES_PATH = os.path.join(ROOT, "data", "processed_data", "breed_profiles.csv")

INPUT_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ImageNet classes 151-268 are domestic dog breeds; 269-275 are wolves, foxes
# and wild dogs, which still count as "a dog-like animal is in the photo".
CANID_RANGE = range(151, 276)

# Girth/length (inches) per cube-root-pound for a "standard" build, and
# per-build multipliers. Calibrated against published breed measurements
# (e.g. a 65 lb Labrador: ~30" chest, ~18" neck, ~22" back).
BASE = {"chest": 7.4, "neck": 4.6, "back": 5.6}
BUILD = {
    "standard":   {"chest": 1.00, "neck": 1.00, "belly": 0.82, "back": 1.00},
    "sighthound": {"chest": 1.05, "neck": 0.85, "belly": 0.62, "back": 1.05},
    "stocky":     {"chest": 1.12, "neck": 1.15, "belly": 0.92, "back": 0.90},
    "long":       {"chest": 1.05, "neck": 1.00, "belly": 0.85, "back": 1.30},
}


def load_profiles(path=PROFILES_PATH):
    profiles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            profiles[int(row["class_index"])] = {
                "name": row["name"],
                "min_lbs": float(row["min_lbs"]),
                "max_lbs": float(row["max_lbs"]),
                "build": row["build"],
                "coat": row["coat"],
                "brachycephalic": row["brachycephalic"] == "yes",
            }
    return profiles


PROFILES = load_profiles()
PROFILES_BY_NAME = {p["name"].lower(): p for p in PROFILES.values()}


def measurements_for_weight(weight_lbs, build="standard"):
    """Estimate neck, chest, belly girth and back length (inches) for a weight."""
    k = BUILD.get(build, BUILD["standard"])
    cube = max(weight_lbs, 1.0) ** (1 / 3)
    chest = BASE["chest"] * cube * k["chest"]
    return {
        "neck": round(BASE["neck"] * cube * k["neck"], 1),
        "chest": round(chest, 1),
        "belly": round(chest * k["belly"], 1),
        "back": round(BASE["back"] * cube * k["back"], 1),
    }


def estimate_body(breeds):
    """Blend breed profiles (weighted by probability) into a body estimate.

    `breeds` is a list of (profile, weight) pairs. Weights are averaged in log
    space so a 50/50 Chihuahua/Labrador guess lands on a medium dog.
    """
    total = sum(w for _, w in breeds)
    log_lo = sum(w * math.log(p["min_lbs"]) for p, w in breeds) / total
    log_hi = sum(w * math.log(p["max_lbs"]) for p, w in breeds) / total
    lo, hi = math.exp(log_lo), math.exp(log_hi)
    mid = math.sqrt(lo * hi)
    build = max(breeds, key=lambda pw: pw[1])[0]["build"]
    return {
        "weight_lbs": round(mid),
        "weight_range": [round(lo), round(hi)],
        "build": build,
        "measurements": measurements_for_weight(mid, build),
        "measurement_ranges": {
            key: [measurements_for_weight(lo, build)[key], measurements_for_weight(hi, build)[key]]
            for key in ("neck", "chest", "belly", "back")
        },
    }


def cpu_limit():
    """CPUs this process may actually use (container cgroup quota, else core count).

    os.cpu_count() reports the host's cores inside a container; sizing the
    ONNX Runtime thread pool from it oversubscribes a CPU-limited container.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def preprocess(image):
    """Resize shorter side to 384, center-crop, normalize -> NCHW float32."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = INPUT_SIZE / min(w, h)
    image = image.resize((max(INPUT_SIZE, round(w * scale)), max(INPUT_SIZE, round(h * scale))), Image.BILINEAR)
    w, h = image.size
    left, top = (w - INPUT_SIZE) // 2, (h - INPUT_SIZE) // 2
    image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return array.transpose(2, 0, 1)[np.newaxis]


class BreedPredictor:
    """Lazily-loaded, thread-safe wrapper around the ONNX session."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._session = None
        self._lock = threading.Lock()

    @property
    def session(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort

                    options = ort.SessionOptions()
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    options.intra_op_num_threads = int(os.environ.get("ORT_THREADS") or min(cpu_limit(), 8))
                    options.inter_op_num_threads = 1
                    self._session = ort.InferenceSession(self.model_path, options, providers=["CPUExecutionProvider"])
                    log.info("ONNX Runtime: %d intra-op threads (cpu_count=%s, cpu_limit=%d)",
                             options.intra_op_num_threads, os.cpu_count(), cpu_limit())
        return self._session

    def analyze(self, image, top_k=3):
        """Return breed guesses, a dog-presence score and a body estimate."""
        session = self.session
        start = time.perf_counter()
        batch = preprocess(image)
        prepped = time.perf_counter()
        probs = session.run(None, {"image": batch})[0][0]
        log.info("analyze timing: preprocess %.0f ms, inference %.0f ms",
                 (prepped - start) * 1000, (time.perf_counter() - prepped) * 1000)
        dog_score = float(probs[CANID_RANGE.start:CANID_RANGE.stop].sum())

        breed_probs = {idx: float(probs[idx]) for idx in PROFILES}
        breed_total = sum(breed_probs.values())
        ranked = sorted(breed_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        breeds = [
            {"name": PROFILES[idx]["name"], "confidence": round(p / breed_total, 3)}
            for idx, p in ranked
        ]

        # Blend the top guesses that carry real weight (mixed-breed friendly).
        blend = [(PROFILES[idx], p) for idx, p in ranked if p / breed_total >= 0.15] or [(PROFILES[ranked[0][0]], 1.0)]
        top = PROFILES[ranked[0][0]]
        return {
            "is_dog": dog_score >= 0.2,
            "dog_score": round(dog_score, 3),
            "breeds": breeds,
            "traits": {"brachycephalic": top["brachycephalic"], "coat": top["coat"], "build": top["build"]},
            "estimate": estimate_body(blend),
        }


predictor = BreedPredictor()
