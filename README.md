
![fixed_smaller](https://github.com/user-attachments/assets/48a7ae47-bda4-47b8-bd6f-a3cb5bc0e0b0)

# DogArmor: Smart AI-Powered Dog Gear Recommender

> 🐶 **DogArmor** is currently in it's post-MVP pre-Final Product phase. We are getting user feedback and still working on the AI models. If you're excited to use our technology, consider giving our github page a ⭐
>  
> Please note development is being done every day!

🚀 [Overview](#overview) | ⚙️ [How It Works](#how-it-works) | 🧠 [Features](#features) | 🛠️ [Tech Stack](#tech-stack-and-contact-info) | 🤝 [Contributing](CONTRIBUTING.md)

## What are we?

DogArmor is an intelligent web app that recommends the perfect collar, harness, and leash combo for your dog utilizing:
* Visual information -- AI will analyze a picture of the dog, looking for anatomical structure, size, and more
* Behavioral information -- the dog's habits and behaviors 
* Environmental information -- the dog's surroundings 
* Budget 

![image](https://github.com/user-attachments/assets/363b32ba-6898-4cee-a972-f00fb6c836eb)

## Overview

🟡 **Why Fit Matters:** 

A properly fitting dog harness is **essential for your dog's comfort, safety, and control** during walks and activities.

- **Too tight?** Can cause chafing, skin irritation, or restrict movement/breathing.
- **Too loose?** Increases the chance of escape, snagging on objects, or uneven pressure that causes injury.

🔴 **The Problem:**

There are countless options—back-clip, front-clip, dual-clip, step-in—plus materials like nylon, mesh, leather, neoprene.  
How do you choose the right one for *your* dog?

🟢 **Our Solution:**

DogArmor makes it simple.

You answer a few short questions about your dog's **behavior**, **environment**, and **budget**, then upload a couple of photos.  
Our AI analyzes everything—**anatomy**, **size**, **habits**—and recommends a tailored gear combo (harness, collar, and leash included).

## How It Works

**1. Upload a photo of your dog to help us understand its size and shape.**

![image](https://github.com/user-attachments/assets/8e260bf3-4c46-4479-bd08-382571174ab6)

**2. Fill out a short form about your dog’s behavior, habits, and environment.**

   ![image](https://github.com/user-attachments/assets/8ad609fb-4154-4b1c-9076-510db72d39fc)

**3. Get instant gear recommendations based on both the photo and your answers.**

![image](https://github.com/user-attachments/assets/0ab8d5ad-1b31-4c54-acc7-8332b0651d2a)

**4. View product links and details tailored to your dog’s unique profile.** -- TO BE IMPLEMENTED 

## Features

🐾 **Anatomy Detection** — Our model analyzes key anatomical points (neck, chest, belly, back length) using pose estimation techniques to determine fit and comfort levels.

🎯 **Behavior Personalization** — Users can input behavioral traits like pulling tendency, aggression, or training level. These parameters are combined with anatomy data to personalize gear suggestions.

🛒 **Intelligent Gear Matching** — Our platform pulls product data from online retailers (e.g., Amazon, Chewy) and cross-references it with your dog’s profile to recommend the most suitable harness, collar, and leash combinations.

🖥️ **Easy-to-Use Interface** — A clean, intuitive web interface powered by Flask and Tailwind CSS, optimized for mobile and desktop, makes uploading photos and receiving gear suggestions fast and frictionless.

🧩 **Future-Proof Backend** — Modular Flask API architecture designed for scalability, with plans to incorporate multiple image angles, user feedback loops, and model fine-tuning.

🔄 **End-to-End Pipeline** — From image input to final product recommendation, the backend handles preprocessing, pose inference, behavior integration, and response generation—all automatically.

## Tech Stack and Contact Info

Built with:
- Python + Flask, HTML/CSS + Tailwind + AOS, SQLite, Future: ML models utilizing DeepLabCut and YOLO 
  
Hosted Using:
- Azure's cloud
  
For all questions, concerns, and business inquiries, please contact us at: dogarmorteam@gmail.com


