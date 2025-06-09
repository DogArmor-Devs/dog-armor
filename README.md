DogArmor is a Smart Dog Gear Recommender.
An intelligent web app that recommends the perfect collar, harness, and leash combo for your dog utilizing the intersection of both visual information from a picture of the dog and behavioral information provided by the user.

![image](https://github.com/user-attachments/assets/363b32ba-6898-4cee-a972-f00fb6c836eb)

## Overview

⚠️ **A properly fitting dog harness is crucial for comfort, saftey, and control during walks and activities.** ⚠️

❌ Too tight? It can cause chafing, skin irritation, or even restrict movement or breathing. 

❌ Too lose? Your dog is more likely to escape, get caught on objects, or suffer pain from uneven pressure distribution.

You might hear options such as back-clip, front-clip, dual-clip, step-in, and materials such as nylon, mesh, leather, neoprene.
So, how are you supposed to find the perfect harness when there are so many options available? Well, our software might just be the solution.

You'll be asked basic questions pertaining to your dogs behavior, it's environment, and your budget. Then, our AI will use those answers in combination with visual information derived from a few pictures of your dog and recommend the perfect combination
of gear (leash and collar are included!).

---

## How It Works

1. Upload a photo of your dog to help us understand its size and shape.
2. Fill out a short form about your dog’s behavior, habits, and environment.
3. Get instant gear recommendations based on both the photo and your answers.
4. View product links and details tailored to your dog’s unique profile.

---

## Features

**Anatomy Detection** - Our model analyzes key anatomical points (neck, chest, belly, back length) using pose estimation techniques to determine fit and comfort levels.

**Behavior Personalization** - Users can input behavioral traits like pulling tendency, aggression, or training level. These parameters are combined with anatomy data to personalize gear suggestions.

**Intelligent Gear Matching** - Our platform pulls product data from online retailers (e.g., Amazon, Chewy) and cross-references it with your dog’s profile to recommend the most suitable harness, collar, and leash combinations.

**Easy-to-Use Interface** - A clean, intuitive web interface powered by Flask and Tailwind CSS, optimized for mobile and desktop, makes uploading photos and receiving gear suggestions fast and frictionless.

**Future-Proof Backend** - Modular Flask API architecture designed for scalability, with plans to incorporate multiple image angles, user feedback loops, and model fine-tuning.

**End-to-End Pipeline** - From image input to final product recommendation, the backend handles preprocessing, pose inference, behavior integration, and response generation—all automatically.

---

## Extra Information

Built with:
- Python + Flask, HTML/CSS + Tailwind + AOS, SQLite, Future: ML models utilizing DeepLabCut and YOLO 
  
Hosted Using:
- Azure's cloud
  
For all questions, concerns, and business inquiries, please contact us at: dogarmorteam@gmail.com


