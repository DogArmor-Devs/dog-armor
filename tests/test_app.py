import io
import os

import pytest
from PIL import Image

os.environ.setdefault("WARM_MODEL", "0")

from app import create_app  # noqa: E402
from src.features.breed_predictor import PROFILES, estimate_body, measurements_for_weight  # noqa: E402
from src.features.gear_recommender import BUDGETS, CATALOG, parse_profile, recommend  # noqa: E402

BASE = {
    "breed": "Labrador Retriever", "weight_lbs": 65, "neck": 18, "chest": 30,
    "pulling": "no", "aggression": "low", "training": "intermediate", "activity": "moderate",
    "working": "no", "sports": "no", "growing": "no", "climate": "moderate", "heat_sensitive": "no",
    "hiking": "no", "hunting": "no", "water": "no", "cleaning_experience": "yes", "budget": "medium",
}


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def rec(**overrides):
    return recommend(parse_profile({**BASE, **overrides}))


@pytest.mark.parametrize("path", ["/", "/demo", "/how-it-works", "/company", "/faq", "/healthz"])
def test_pages_render(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("old, new", [("/about", "/company"), ("/team", "/company"), ("/why-us", "/company"), ("/features", "/")])
def test_old_pages_redirect(client, old, new):
    res = client.get(old)
    assert res.status_code == 301 and res.headers["Location"].endswith(new)


def test_no_dashes_in_copy(client):
    for path in ["/", "/demo", "/how-it-works", "/company", "/faq"]:
        html = client.get(path).get_data(as_text=True)
        assert "—" not in html and "–" not in html, path


def test_unknown_page_404(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/api/nope").get_json()["status"] == "error"


def test_profiles_cover_all_imagenet_dog_classes():
    assert sorted(PROFILES) == list(range(151, 269))
    assert all(p["min_lbs"] <= p["max_lbs"] for p in PROFILES.values())


def test_measurement_estimates_are_plausible():
    lab = measurements_for_weight(65)
    assert 28 <= lab["chest"] <= 32 and 16 <= lab["neck"] <= 20
    chihuahua = estimate_body([(PROFILES[151], 1.0)])
    assert chihuahua["weight_lbs"] <= 6 and chihuahua["measurements"]["chest"] < 14


def test_analyze_rejects_non_dog_image(client):
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (40, 120, 200)).save(buf, "JPEG")
    buf.seek(0)
    data = client.post("/api/analyze", data={"image": (buf, "sky.jpg")}, content_type="multipart/form-data").get_json()
    assert data["status"] == "success"
    assert data["is_dog"] is False
    assert len(data["breeds"]) == 3


def test_analyze_rejects_garbage(client):
    res = client.post("/api/analyze", data={"image": (io.BytesIO(b"not an image"), "x.jpg")}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_estimate_endpoint(client):
    data = client.get("/api/estimate?breed=pug").get_json()
    assert data["traits"]["brachycephalic"] is True
    data = client.get("/api/estimate?weight=40&breed=Whippet").get_json()
    assert data["estimate"]["build"] == "sighthound"
    assert client.get("/api/estimate?breed=dragon").status_code == 400


def test_recommend_endpoint_validates(client):
    res = client.post("/api/recommend", json={**BASE, "aggression": "extreme"})
    assert res.status_code == 400 and "aggression" in res.get_json()["message"]
    res = client.post("/api/recommend", json=BASE)
    body = res.get_json()
    assert body["status"] == "success"
    assert set(body["combo"]) == {"harness", "collar", "leash"}


def test_puller_gets_front_clip_harness():
    assert "front-clip" in rec(pulling="yes")["combo"]["harness"]["tags"]


def test_brachycephalic_never_gets_head_halter():
    result = recommend(parse_profile({**BASE, "breed": "Pug", "weight_lbs": 16, "neck": 13, "chest": 21,
                                      "pulling": "yes", "brachycephalic": True}))
    assert "head-halter" not in result["combo"]["collar"]["tags"]
    assert any("Flat-faced" in note for note in result["notes"])


def test_reactive_dog_gets_control_leash():
    result = rec(aggression="high")
    assert "double-handle" in result["combo"]["leash"]["tags"]
    assert "handle" in result["combo"]["harness"]["tags"]


def test_water_dog_gets_waterproof_gear_and_life_jacket():
    result = rec(water="yes")
    assert "waterproof" in result["combo"]["leash"]["tags"]
    assert "leather" not in result["combo"]["collar"]["tags"]
    assert any("water" in e["tags"] for e in result["extras"])


def test_heat_sensitive_dog_gets_breathable_harness():
    assert "breathable" in rec(heat_sensitive="yes")["combo"]["harness"]["tags"]


def test_small_dog_fits():
    result = rec(breed="Chihuahua", weight_lbs=5, neck=8, chest=13)
    harness = next(p for p in CATALOG if p["id"] == result["combo"]["harness"]["id"])
    assert harness["fit"]["chest"][0] - 1 <= 13 <= harness["fit"]["chest"][1] + 1
    assert "head-halter" not in result["combo"]["collar"]["tags"]


@pytest.mark.parametrize("budget", list(BUDGETS))
def test_budget_caps_respected(budget):
    result = rec(budget=budget, hiking="yes", pulling="yes")
    for category, item in result["combo"].items():
        cap = BUDGETS[budget][category]
        assert cap is None or item["price"] <= cap


def test_links_point_to_product_pages():
    for item in rec()["combo"].values():
        assert item["links"]
        assert all(link["url"].startswith(("https://www.amazon.com/dp/", "https://www.chewy.com/")) for link in item["links"])
    for product in CATALOG:
        urls = product["links"].values()
        assert urls, product["id"]
        assert all("/dp/" in url and "/s?" not in url for url in urls), product["id"]


def test_only_chosen_gear_is_recommended():
    result = rec(gear=["harness"])
    assert list(result["combo"]) == ["harness"] and list(result["alternatives"]) == ["harness"]
    assert result["total_price"] == result["combo"]["harness"]["price"]
    assert list(rec(gear=["leash", "harness"])["combo"]) == ["harness", "leash"]


def test_budget_label_matches_chosen_gear():
    assert rec(gear=["harness"], budget="low")["budget"]["label"] == "under $30"
    assert rec(gear=["harness", "collar"], budget="medium")["budget"]["label"] == "$45 to $90"
    assert rec(budget="medium")["budget"]["label"] == "$65 to $130"


@pytest.mark.parametrize("gear", [[], ["saddle"], "nope"])
def test_invalid_gear_rejected(client, gear):
    res = client.post("/api/recommend", json={**BASE, "gear": gear})
    assert res.status_code == 400 and "gear" in res.get_json()["message"]


def test_demo_page_exposes_budget_caps(client):
    html = client.get("/demo").get_data(as_text=True)
    assert 'name="gear" value="harness"' in html and "data-caps=" in html
    assert sum(BUDGETS["low"][c] for c in ("harness", "collar", "leash")) == 65
