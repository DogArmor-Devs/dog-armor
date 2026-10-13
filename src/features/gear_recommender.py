"""Turn a dog profile (body + behavior + environment + budget) into a gear combo.

Every answer in the questionnaire adds weighted preferences ("wants") or hard
exclusions for product tags, per gear category. Products that don't fit the
dog's measurements or the per-item budget are filtered out, the rest are scored
by summing the weights of their tags, and the best harness, collar and leash
are returned with the reasons that drove each pick.
"""

import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG_PATH = os.path.join(ROOT, "data", "processed_data", "gear_catalog.json")

CATEGORIES = ("harness", "collar", "leash")
LEVELS = ("low", "moderate", "high")
TRAINING_LEVELS = ("beginner", "intermediate", "advanced")
CLIMATES = ("hot", "moderate", "cold")
BUDGETS = {
    # per-item caps in USD; the tier's total range is the sum over the chosen gear
    "low": {"harness": 30, "collar": 15, "leash": 20, "extra": 30},
    "medium": {"harness": 60, "collar": 30, "leash": 40, "extra": 60},
    "high": {"harness": None, "collar": None, "leash": None, "extra": None},
}
YES_NO_FIELDS = ("pulling", "hiking", "hunting", "working", "water", "sports", "growing", "cleaning_experience", "heat_sensitive")

RETAILERS = {"amazon": "Amazon", "chewy": "Chewy"}

HARNESS_SIZES = [("XXS", 9, 13), ("XS", 13, 17), ("S", 17, 22), ("M", 22, 27), ("L", 27, 32), ("XL", 32, 46)]
COLLAR_SIZES = [("XS", 8, 11), ("S", 11, 14), ("M", 14, 18), ("L", 18, 22), ("XL", 22, 30)]


def budget_label(tier, gear):
    """Human-readable total range for a budget tier over the chosen gear."""
    low = sum(BUDGETS["low"][c] for c in gear)
    if tier == "low":
        return f"under ${low}"
    if tier == "medium":
        return f"${low} to ${sum(BUDGETS['medium'][c] for c in gear)}"
    return "no limit"


def load_catalog(path=CATALOG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["products"]


CATALOG = load_catalog()


class ProfileError(ValueError):
    pass


def parse_profile(data):
    """Validate and normalize the questionnaire payload."""
    def number(key, low, high, required=True):
        value = data.get(key)
        if value in (None, ""):
            if required:
                raise ProfileError(f"'{key}' is required")
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ProfileError(f"'{key}' must be a number")
        if not low <= value <= high:
            raise ProfileError(f"'{key}' must be between {low} and {high}")
        return value

    def choice(key, options):
        value = str(data.get(key, "")).strip().lower()
        if value not in options:
            raise ProfileError(f"'{key}' must be one of: {', '.join(options)}")
        return value

    def gear_list():
        value = data.get("gear", list(CATEGORIES))
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, (list, tuple)):
            raise ProfileError("'gear' must be a list")
        chosen = {str(v).strip().lower() for v in value}
        if not chosen or not chosen <= set(CATEGORIES):
            raise ProfileError(f"'gear' must include at least one of: {', '.join(CATEGORIES)}")
        return [c for c in CATEGORIES if c in chosen]

    def yes_no(key):
        value = data.get(key)
        if isinstance(value, bool):
            return value
        value = str(value or "").strip().lower()
        if value not in ("yes", "no"):
            raise ProfileError(f"'{key}' must be yes or no")
        return value == "yes"

    profile = {
        "breed": str(data.get("breed") or "").strip()[:80],
        "weight_lbs": number("weight_lbs", 1, 250),
        "neck": number("neck", 5, 40),
        "chest": number("chest", 8, 60),
        "belly": number("belly", 6, 60, required=False),
        "back": number("back", 4, 50, required=False),
        "aggression": choice("aggression", LEVELS),
        "activity": choice("activity", LEVELS),
        "training": choice("training", TRAINING_LEVELS),
        "climate": choice("climate", CLIMATES),
        "budget": choice("budget", tuple(BUDGETS)),
        "gear": gear_list(),
        "brachycephalic": bool(data.get("brachycephalic")),
        "build": data.get("build") if data.get("build") in ("standard", "sighthound", "stocky", "long") else "standard",
    }
    for key in YES_NO_FIELDS:
        profile[key] = yes_no(key)
    return profile


class Preferences:
    def __init__(self):
        self.weights = {c: defaultdict(float) for c in CATEGORIES + ("extra",)}
        self.reasons = {c: defaultdict(list) for c in CATEGORIES + ("extra",)}
        self.excluded = {c: set() for c in CATEGORIES + ("extra",)}

    def want(self, category, tag, weight, reason=None):
        self.weights[category][tag] += weight
        if reason and weight > 0:
            self.reasons[category][tag].append((weight, reason))

    def exclude(self, category, tag):
        self.excluded[category].add(tag)


def build_preferences(p):
    prefs = Preferences()
    want = prefs.want
    small = p["weight_lbs"] < 15
    toy = p["weight_lbs"] < 12
    large = p["weight_lbs"] >= 70
    hot = p["heat_sensitive"] or p["climate"] == "hot"

    # Baseline: an ID collar and a regular leash.
    want("collar", "flat", 1.0)
    want("leash", "standard", 1.0)
    want("harness", "padded", 0.5, "Padding spreads pressure so straps don't rub.")
    want("harness", "adjustable", 0.5, "Multiple adjustment points for a precise fit.")

    if p["pulling"]:
        want("harness", "front-clip", 4, "Front clip turns your dog sideways when it pulls. The most effective harness fix for pulling.")
        want("leash", "padded-handle", 1, "Padded handle protects your hands from sudden pulls.")
        want("leash", "bungee", 0.5, "Shock-absorbing section softens jolts.")
        want("leash", "hands-free", -2)
        if not p["brachycephalic"] and not toy:
            want("collar", "martingale", 1.5, "Limited-slip design gives gentle feedback without choking.")
            if p["weight_lbs"] >= 30 and p["training"] != "advanced":
                want("collar", "head-halter", 2.5, "Head halter gives you steering on a strong puller. Introduce it slowly.")

    if p["aggression"] == "high":
        want("harness", "handle", 3, "Grab handle lets you hold your dog close in tense situations.")
        want("harness", "escape-resistant", 2, "Escape-resistant fit keeps a reactive dog secure.")
        want("harness", "dual-clip", 1, "Two leash points give you extra control.")
        want("collar", "martingale", 3, "Martingale can't be backed out of during a lunge.")
        want("leash", "double-handle", 4, "Second handle near the clip gives instant close control when meeting other dogs.")
        want("leash", "traffic-handle", 2)
        prefs.exclude("leash", "hands-free")
        prefs.exclude("leash", "long-line")
        want("leash", "bungee", -3)
    elif p["aggression"] == "moderate":
        want("harness", "handle", 1.5, "Handle for quick control when needed.")
        want("leash", "double-handle", 2, "Second handle for close control in busy places.")
        want("leash", "hands-free", -1)

    if p["activity"] == "high":
        want("harness", "dual-clip", 1, "Versatile clip options for different activities.")
        want("harness", "durable", 1, "Built to hold up to a high-energy dog.")
        want("harness", "lightweight", 1, "Lightweight so it won't slow an active dog down.")
        if p["aggression"] == "low" and not p["pulling"]:
            want("leash", "hands-free", 2, "Hands-free option for running together.")
            want("leash", "bungee", 2, "Bungee section absorbs shock at speed.")
        want("leash", "durable", 1, "Durable enough for daily heavy use.")
    elif p["activity"] == "low":
        want("harness", "lightweight", 1, "Lightweight and easy to wear for relaxed walks.")
        want("harness", "step-in", 0.5, "Easy to put on.")

    if p["hiking"]:
        want("harness", "handle", 2, "Handle lets you help your dog over rocks and logs on the trail.")
        want("harness", "hiking", 2, "Designed for trail use.")
        want("harness", "durable", 1, "Tough enough for the trail.")
        want("leash", "hiking", 2, "Trail-ready leash.")
        want("leash", "durable", 1)
    if p["hunting"]:
        want("harness", "reflective", 2, "High visibility in the field.")
        want("harness", "durable", 2, "Tough materials for brush and briars.")
        want("leash", "long-line", 4, "Long line for field training and recall work.")
        want("leash", "hunting", 2)
        want("collar", "reflective", 1.5, "Keeps your dog visible in the field.")
    if p["working"]:
        want("harness", "working", 4, "Working-dog harness built for service and protection roles.")
        want("harness", "handle", 3, "Rigid handle for close guidance.")
        want("harness", "durable", 2, "Heavy-duty construction.")
        want("collar", "durable", 2, "Heavy-duty collar.")
        want("collar", "working", 1.5)
        want("leash", "double-handle", 2, "Two handles for working control.")
        want("leash", "durable", 2)
    if p["water"]:
        want("harness", "quick-dry", 2, "Dries quickly after swims.")
        want("harness", "padded", -1)
        want("collar", "waterproof", 3, "Waterproof coating won't get soggy or smelly.")
        want("leash", "waterproof", 4, "Waterproof material that stays clean after swims.")
        want("collar", "leather", -4)
        want("leash", "leather", -4)
    if p["sports"]:
        want("harness", "lightweight", 2, "Lightweight for agility and sports.")
        want("harness", "dual-clip", 1)
    if p["growing"]:
        want("harness", "adjustable", 3, "Lots of adjustment to grow with your puppy.")
        want("collar", "lightweight", 1, "Light and comfortable for a puppy.")
    if not p["cleaning_experience"]:
        for category in CATEGORIES:
            want(category, "easy-clean", 1, "Low-maintenance and machine-washable or wipe-clean.")
            want(category, "leather", -2.5)
    if hot:
        want("harness", "breathable", 3, "Breathable mesh helps a heat-sensitive dog stay cool.")
        want("harness", "mesh", 1)
        want("harness", "lightweight", 1, "Less material means less trapped heat.")
        want("harness", "padded", -1.5)
    if p["climate"] == "cold":
        want("harness", "padded", 1, "Padded straps add a bit of warmth in cold weather.")
        want("harness", "reflective", 1, "Reflective trim for short winter days.")
        want("leash", "reflective", 0.5)

    if p["training"] == "beginner":
        want("harness", "front-clip", 1, "Front clip makes training walks easier.")
        want("collar", "head-halter", -1)
    elif p["training"] == "advanced":
        want("harness", "lightweight", 0.5)

    # Body-based rules.
    if small:
        want("harness", "small-dog", 2, "Sized and shaped for small dogs.")
        want("harness", "step-in", 1.5, "Step-in design, nothing pulled over the head.")
        want("harness", "lightweight", 1.5)
        want("leash", "lightweight", 2, "Light clip and leash won't weigh down a small dog.")
    if toy:
        prefs.exclude("collar", "head-halter")
        want("collar", "martingale", -2)
    if large:
        want("harness", "durable", 2, "Heavy-duty hardware for a big, strong dog.")
        want("harness", "handle", 1, "Handle helps you control a big dog.")
        want("leash", "durable", 2, "Rated for a big, strong dog.")
        want("leash", "padded-handle", 1)
        want("harness", "small-dog", -5)
    if p["brachycephalic"]:
        prefs.exclude("collar", "head-halter")
        want("collar", "martingale", -3)
    if p["build"] == "sighthound":
        want("collar", "martingale", 3, "Sighthound heads are narrower than their necks. A martingale stops them slipping out.")
        want("harness", "escape-resistant", 1.5, "Escape-resistant fit for a deep-chested, narrow-waisted dog.")

    # Extras.
    if p["water"]:
        want("extra", "water", 5, "A life jacket is a must for boating and open water.")
    if hot:
        want("extra", "cooling", 3, "Cooling vest for hot days.")
    if p["hunting"] or p["climate"] == "cold":
        want("extra", "visibility", 2, "LED light keeps your dog visible in low light.")
    return prefs


def fits(product, p):
    fit = product["fit"]
    tolerance = 1.0
    if "chest" in fit and not fit["chest"][0] - tolerance <= p["chest"] <= fit["chest"][1] + tolerance:
        return False
    if "neck" in fit and not fit["neck"][0] - tolerance <= p["neck"] <= fit["neck"][1] + tolerance:
        return False
    if "weight" in fit and not fit["weight"][0] <= p["weight_lbs"] <= fit["weight"][1]:
        return False
    if "max_lbs" in fit and p["weight_lbs"] > fit["max_lbs"]:
        return False
    if "min_lbs" in fit and p["weight_lbs"] < fit["min_lbs"]:
        return False
    return True


def score(product, prefs):
    weights = prefs.weights[product["category"]]
    return sum(weights.get(tag, 0.0) for tag in product["tags"])


def explain(product, prefs, limit=3):
    reasons = prefs.reasons[product["category"]]
    picked = sorted(
        ((w, text) for tag in product["tags"] for w, text in reasons.get(tag, [])),
        key=lambda wt: wt[0],
        reverse=True,
    )
    out = []
    for _, text in picked:
        if text not in out:
            out.append(text)
        if len(out) == limit:
            break
    return out


def size_label(value, chart):
    for label, low, high in chart:
        if low <= value < high:
            return label
    return chart[0][0] if value < chart[0][1] else chart[-1][0]


def sizing(product, p):
    category = product["category"]
    if category == "harness" or (category == "extra" and "chest" in product["fit"]):
        return f"Usually size {size_label(p['chest'], HARNESS_SIZES)} for a {p['chest']:.0f} inch chest. Confirm on the {product['brand']} size chart."
    if category == "collar":
        return f"Usually size {size_label(p['neck'], COLLAR_SIZES)} for a {p['neck']:.0f} inch neck. Two fingers should fit underneath."
    if category == "leash":
        if p["weight_lbs"] < 15:
            width = "3/8 inch"
        elif p["weight_lbs"] < 50:
            width = "5/8 or 3/4 inch"
        else:
            width = "1 inch"
        return f"Get the {width} width for a {p['weight_lbs']:.0f} lb dog."
    return None


def links(product):
    """Direct product-page links, Amazon first."""
    return [{"retailer": RETAILERS[r], "url": product["links"][r]} for r in RETAILERS if r in product["links"]]


def present(product, prefs, p, rank_score):
    return {
        "id": product["id"],
        "category": product["category"],
        "brand": product["brand"],
        "name": product["name"],
        "price": product["price"],
        "material": product["material"],
        "summary": product["summary"],
        "tags": product["tags"],
        "score": round(rank_score, 2),
        "reasons": explain(product, prefs),
        "sizing": sizing(product, p),
        "links": links(product),
    }


def rank(category, prefs, p, catalog):
    cap = BUDGETS[p["budget"]][category]
    candidates = [
        prod for prod in catalog
        if prod["category"] == category
        and fits(prod, p)
        and not (set(prod["tags"]) & prefs.excluded[category])
    ]
    within = [prod for prod in candidates if cap is None or prod["price"] <= cap]
    # If nothing fits the budget, fall back to the cheapest options that fit the dog.
    pool = within or sorted(candidates, key=lambda prod: prod["price"])[:3]
    return sorted(((score(prod, prefs), prod) for prod in pool), key=lambda sp: (-sp[0], sp[1]["price"])), bool(within)


def material_note(p):
    if p["heat_sensitive"] or p["climate"] == "hot":
        return "Mesh: lightweight and breathable, ideal for hot weather or dogs prone to overheating."
    if p["water"]:
        return "BioThane or coated webbing: waterproof, odor-resistant and quick to rinse after swims."
    if not p["cleaning_experience"]:
        return "Nylon: durable, affordable and machine-washable. Skip leather, which needs regular conditioning."
    if p["activity"] == "high":
        return "Nylon: durable, lightweight and great for active dogs."
    return "Nylon for everyday durability, or leather if you like a classic look and don't mind conditioning it."


def safety_notes(p):
    notes = []
    if p["brachycephalic"]:
        notes.append("Flat-faced breeds have narrow airways: always attach the leash to the harness, use the collar only for ID tags, and skip head halters.")
    if p["weight_lbs"] < 12:
        notes.append("Small dogs are prone to tracheal collapse, so walk on the harness rather than the collar.")
    if p["build"] == "sighthound":
        notes.append("Sighthounds slip standard collars easily. A martingale or escape-resistant harness keeps them secure.")
    if p["growing"]:
        notes.append("Puppies grow fast. Re-measure every two or three weeks and size up when the straps run out.")
    if p["pulling"] and p["training"] == "beginner":
        notes.append("Gear helps, but pair it with reward-based loose-leash training for lasting results.")
    notes.append("Fit check: two fingers should slide under every strap, and the harness shouldn't ride into the armpits.")
    return notes


def recommend(profile, catalog=None):
    """Return the recommended combo plus alternatives for a parsed profile."""
    catalog = catalog or CATALOG
    prefs = build_preferences(profile)
    picks, alternatives, over_budget = {}, {}, []
    for category in profile["gear"]:
        ranked, within_budget = rank(category, prefs, profile, catalog)
        if not ranked:
            continue
        if not within_budget:
            over_budget.append(category)
        picks[category] = present(ranked[0][1], prefs, profile, ranked[0][0])
        alternatives[category] = [present(prod, prefs, profile, s) for s, prod in ranked[1:3]]

    extras = []
    if any(w > 0 for w in prefs.weights["extra"].values()):
        ranked, _ = rank("extra", prefs, profile, catalog)
        extras = [present(prod, prefs, profile, s) for s, prod in ranked if s > 0][:2]

    total = sum(item["price"] for item in picks.values())
    return {
        "combo": picks,
        "alternatives": alternatives,
        "extras": extras,
        "total_price": total,
        "budget": {"tier": profile["budget"], "label": budget_label(profile["budget"], profile["gear"]), "over_budget": over_budget},
        "gear": profile["gear"],
        "material_note": material_note(profile),
        "notes": safety_notes(profile),
        "measurements": {k: profile[k] for k in ("weight_lbs", "neck", "chest", "belly", "back") if profile.get(k) is not None},
    }
