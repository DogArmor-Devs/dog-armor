<div align="center">

<img src="app/static/dog-armor.png" width="96" alt="DogArmor logo">

# DogArmor

**The right harness, collar and leash for your dog, picked from one photo and a few questions.**

[**Try it live: dogarmor.up.railway.app**](https://dogarmor.up.railway.app)

</div>

![DogArmor home page](docs/images/home.png)

## Watch the demo

A 1:19 walkthrough of the live demo: choosing gear, uploading a photo, adjusting measurements, answering the questions and opening the recommended harness on Amazon.

[![Watch the DogArmor demo video](docs/images/demo-thumbnail.png)](https://cdn.jsdelivr.net/gh/DogArmor-Devs/dog-armor@main/docs/dogarmor-demo.mp4)

<sub>The video file lives in this repo at [`docs/dogarmor-demo.mp4`](docs/dogarmor-demo.mp4); the thumbnail streams it through jsDelivr so it plays in the browser.</sub>

## What it does

- **Recognizes your dog** from a single photo (118 breeds, blended for mixes) and checks there is a dog in the photo at all.
- **Estimates the measurements that decide fit**: weight, neck, chest girth, belly and back length. You can replace any of them with a real measurement.
- **Picks the gear you ask for** (a harness, collar, leash or any mix) that fits those measurements, your dog's behavior, where you walk and your budget, and explains every pick.

No account needed. Photos are analyzed in memory and never stored.

## How to use it

### 1. Choose your gear

Pick a harness, collar, leash or any mix. DogArmor only recommends, and budgets for, what you choose.

![Step 1: choose which gear you want](docs/images/step-1-gear.png)

### 2. Upload a photo

Take it from the side at your dog's eye level, with your dog standing and its face visible. Front-on shots and head close-ups hide the body and make the breed harder to read. DogArmor shows its top three breed matches. Click a different one if it got it wrong.

![Step 2: upload a photo and see the breed matches](docs/images/step-2-photo.png)

### 3. Check the measurements

Measurements are estimated from the breed and its build, not measured from the photo. Fields marked **Estimated** can be overwritten. Enter a real weight and the girths recalculate around it.

![Step 3: estimated measurements you can edit](docs/images/step-3-measurements.png)

### 4. Answer a few questions

Thirteen quick questions about behavior (pulling, reactivity, training, energy), activities (hiking, hunting, water, sports), climate and gear upkeep. Then pick a budget; the price ranges add up only the gear you chose (a harness alone is under $30, $30 to $60, or no limit).

![Step 4: behavior questions](docs/images/step-4-questions.png)

### 5. Get your picks

Each item you asked for comes with the reasons it was chosen, the size to order, two alternatives and links to Amazon and Chewy, plus material and safety notes for your dog.

![Step 5: the recommended harness, collar and leash](docs/images/step-5-results.png)

## How it works

| Stage | What happens |
|---|---|
| **Breed recognition** | EfficientNetV2-S (ImageNet weights) exported to ONNX and run with ONNX Runtime, about 20 ms per photo on CPU. Its non-dog classes double as a "no dog found" check. |
| **Measurements** | Each breed has a typical adult weight range and build (standard, sighthound, stocky, long-backed). Girths and back length scale with the cube root of weight, adjusted for build. Mixed breeds blend the top matches by confidence. |
| **Recommendation** | A curated catalog of 36 products, each tagged with features, material, price and fit range. Answers add weight to the tags that matter for your dog. Products that don't fit or exceed the budget are filtered out, and the highest scoring harness, collar and leash win. |
| **Safety rules** | No head halters for flat-faced or toy breeds, harness-first advice for small dogs, martingales for sighthounds. |

**Accuracy:** on everyday dog photos the model names the exact breed 81% of the time, and the right breed is in its top three 97% of the time.

**Limits:** a single photo has no scale, so measurements are breed-based estimates. For the most precise sizing, measure your dog and enter the numbers. Prices are typical and change often.

## Run it locally

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                  # http://localhost:5000
```

| Task | Command |
|---|---|
| Run tests | `pip install pytest && python -m pytest` |
| Rebuild CSS after editing templates | `npm install && npm run build:css` |
| Regenerate the model file | `pip install torch torchvision onnx onnxscript && python -m src.features.export_onnx` |

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Multipart `image` → breed matches, dog check and measurement estimate |
| `GET /api/estimate?breed=&weight=` | Measurements for a breed and/or known weight |
| `POST /api/recommend` | JSON profile → picks for the requested `gear` (any of `harness`, `collar`, `leash`), alternatives, extras and notes |
| `GET /api/breeds` | All supported breed names |
| `GET /healthz` | Health check |

## Deployment

Production runs on [Railway](https://railway.com) from the `Dockerfile` (gunicorn). Pushes to `main` deploy automatically. Service settings live in `.railway/railway.ts`: preview changes with `railway config plan` and apply them with `railway config apply`. To deploy manually, run `railway up --service web`.

## Project structure

```
app/
  routes.py               pages and JSON API
  templates/              Jinja templates (Tailwind)
  static/                 compiled CSS, demo.js, images
src/features/
  breed_predictor.py      ONNX inference and measurement estimates
  gear_recommender.py     scoring engine
  export_onnx.py          builds models/breed_classifier.onnx
data/processed_data/
  breed_profiles.csv      weight range, build and coat for 118 breeds
  gear_catalog.json       product catalog
models/                   breed_classifier.onnx
tests/                    pytest suite
```

## Team

Built by **Vihaan Kesharwani** (design and frontend) and **Anto Francis** (models and backend).
Questions or business inquiries: [dogarmorteam@gmail.com](mailto:dogarmorteam@gmail.com)

Image and dataset credits are in [Citations.md](Citations.md).
