![fixed_smaller](https://github.com/user-attachments/assets/48a7ae47-bda4-47b8-bd6f-a3cb5bc0e0b0)

# DogArmor: Smart AI-Powered Dog Gear Recommender

> Try it live: **https://dogarmor.up.railway.app**

🚀 [Overview](#overview) | ⚙️ [How It Works](#how-it-works) | 🧠 [Features](#features) | 🛠️ [Tech Stack](#tech-stack-and-contact-info) | 💻 [Run It Yourself](#run-it-yourself)

## What are we?

DogArmor is an intelligent web app that recommends the perfect collar, harness, and leash combo for your dog using:
* Visual information: AI analyzes a picture of the dog to identify its breed and build and estimate its size and measurements
* Behavioral information: the dog's habits and behaviors
* Environmental information: the dog's surroundings and activities
* Budget

![image](https://github.com/user-attachments/assets/363b32ba-6898-4cee-a972-f00fb6c836eb)

## Overview

🟡 **Why Fit Matters:**

A properly fitting dog harness is **essential for your dog's comfort, safety, and control** during walks and activities.

- **Too tight?** Can cause chafing, skin irritation, or restrict movement/breathing.
- **Too loose?** Increases the chance of escape, snagging on objects, or uneven pressure that causes injury.

🔴 **The Problem:**

There are countless options (back-clip, front-clip, dual-clip, step-in) plus materials like nylon, mesh, leather, and BioThane.
How do you choose the right one for *your* dog?

🟢 **Our Solution:**

DogArmor makes it simple.

Upload a photo of your dog and answer a few short questions about its **behavior**, **environment**, and **your budget**.
Our AI analyzes everything (**anatomy**, **size**, **habits**) and recommends a tailored gear combo: harness, collar, and leash.

## How It Works

1. Upload a photo of your dog to help us understand its size and shape.

![image](https://github.com/user-attachments/assets/8e260bf3-4c46-4479-bd08-382571174ab6)

2. Fill out a short form about your dog's behavior, habits, and environment.

![image](https://github.com/user-attachments/assets/8ad609fb-4154-4b1c-9076-510db72d39fc)

3. Get instant gear recommendations based on both the photo and your answers.

![image](https://github.com/user-attachments/assets/0ab8d5ad-1b31-4c54-acc7-8332b0651d2a)

4. View product links and details tailored to your dog's unique profile: the size to buy, why each item was picked, and links to Amazon and Chewy.

## Features

🐾 **Anatomy Estimation**: An image model identifies your dog's breed (118 breeds, blending the top matches for mixed breeds) and checks that the photo actually contains a dog. From the breed's typical adult size and build (standard, deep-chested sighthound, stocky, long-backed) DogArmor estimates **weight, neck, chest, belly and back length**. Owners can override any number with a tape-measured value, and the rest re-estimate from it.

🎯 **Behavior Personalization**: Pulling, aggression/reactivity, training level, activity level, working-dog and sport use, and puppy growth all change what gear is recommended (front-clip harnesses for pullers, double-handle leashes and martingales for reactive dogs, adjustable gear for puppies, and so on).

🌦️ **Environment Matching**: Climate, heat sensitivity (auto-suggested for flat-faced and thick-coated breeds), hiking, hunting, water activities and how much gear maintenance you're comfortable with.

🛒 **Intelligent Gear Matching**: A curated catalog of harnesses, collars, leashes and extras from brands like Ruffwear, PetSafe, Kurgo and Julius-K9, each tagged with features, materials, price and fit range. Products are filtered by your dog's measurements and your budget, scored against your answers, and returned with the reasons they were picked, a size suggestion, alternatives, and Amazon/Chewy links.

🛡️ **Safety-Aware**: Breed-specific safety rules, such as no head halters for brachycephalic breeds, harness-first advice for toy breeds prone to tracheal collapse, and martingales for sighthounds.

🖥️ **Easy-to-Use Interface**: A clean, responsive interface powered by Flask and Tailwind CSS, optimized for mobile and desktop.

🧩 **Modular Flask API**: `/api/analyze` (photo → breed + measurements), `/api/estimate` (breed/weight → measurements), `/api/recommend` (profile → gear combo).

🔄 **End-to-End Pipeline**: From image input to final product recommendation, the backend handles preprocessing, inference, measurement estimation, behavior integration, and response generation automatically. Photos are processed in memory and never stored.

### About the model

The breed classifier is torchvision's EfficientNetV2-S (ImageNet weights) exported to ONNX and served with ONNX Runtime (~20 ms per photo on CPU, no PyTorch needed in production). We benchmarked it against our earlier ResNet-18 fine-tuned on Stanford Dogs:

| Model | Held-out Stanford Dogs | Real-world photos | Real-world top-3 |
|---|---|---|---|
| Fine-tuned ResNet-18 (previous) | 55% | 34% | 61% |
| EfficientNetV2-S (current) | 97% | 81% | 97% |

Measurements are estimates derived from breed morphometrics, not direct pose estimation, so entering real measurements always gives the most precise sizing.

## Tech Stack and Contact Info

Built with:
- Python + Flask, ONNX Runtime, HTML/CSS + Tailwind + AOS, vanilla JS

Hosted using:
- Railway (Docker)

For all questions, concerns, and business inquiries, please contact us at: dogarmorteam@gmail.com

## Run It Yourself

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                                         # http://localhost:5000
```

Run the tests with `pip install pytest && python -m pytest`.

After changing templates or CSS classes, rebuild the stylesheet: `npm install && npm run build:css`.

To regenerate the model file (needs `torch torchvision onnx onnxscript`): `python -m src.features.export_onnx`.

### Deploying to Railway

Production runs on Railway from the `Dockerfile` (gunicorn, health check on `/healthz`). Service settings live in code in `.railway/railway.ts` (Railway Infrastructure as Code): preview changes with `railway config plan` and apply them with `railway config apply`. The `PORT=8080` variable is set in the Railway dashboard.

To deploy the current code manually: `railway up --service web`. Pushes to `main` redeploy automatically once the Railway GitHub App is installed on the DogArmor-Devs organization.

### Project layout

```
app/                    Flask app (routes, templates, static assets, demo.js)
src/features/           breed_predictor.py (ONNX inference + measurements),
                        gear_recommender.py (scoring engine), export_onnx.py
data/processed_data/    breed_profiles.csv, gear_catalog.json
models/                 breed_classifier.onnx
tests/                  pytest suite
```
