from flask import Flask, request, jsonify, render_template
import random
import logging
import pandas as pd
from datetime import datetime
import os
from werkzeug.utils import secure_filename

# Import for ML
import torch
from torchvision import transforms, models
from PIL import Image
import json
from src.features.breed_predictor import predict_breed

app = Flask(__name__, static_folder='static', template_folder='templates')

# A folder inside static called 'uploads' for storing 
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transform.ToTensor(),
])


# 🧾 Logs user gear recommendation requests
logging.basicConfig(filename='gear_requests.log', level=logging.INFO)

# 📄 Load CSV of gear options (you’ll create this later)
gear_data = pd.read_csv('gear_data.csv')

# 🌐 Web page routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/how-it-works') 
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/why-us')
def why_us():
    return render_template('why-us.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()

    # Extract user input
    breed = data.get("breed", "").lower()
    weight = data.get("weight", "").lower()
    chest = data.get("chest_bridge_length", "").lower()
    neck = data.get("neck_circumference", "").lower()
    back = data.get("back_bridge_length", "").lower()
    belly = data.get("belly_circumference", "").lower()
    age = data.get("age", "").lower()
    pull = data.get("pull", "").lower()
    guard_dog = data.get("guard_dog", "").lower()
    total_dogs = data.get("total_dogs", "").lower()
    budget = data.get("budget", "").lower()

    # Very basic filtering: find matching gear row
    filtered = gear_data[
        (gear_data["breed"].str.lower().str.contains(breed, na=False)) |
        (gear_data["weight"].str.lower().str.contains(weight, na=False)) |
        (gear_data["neck_circumference"].str.lower().str.contains(neck, na=False)) |
        (gear_data["pull"].str.lower().str.contains(pull, na=False)) |
        (gear_data["budget"].str.lower().str.contains(budget, na=False))
    ]

    if filtered.empty:
        recommendation = gear_data.sample(1).to_dict(orient="records")[0]
    else:
        recommendation = filtered.sample(1).to_dict(orient="records")[0]


    # 🧾 Log this request
    logging.info(f"[{datetime.now()}] /recommend requested with data: {data}")

    return jsonify({
        "recommendations": recommendation,
        "input": data
    })


@app.route('/behavior-gear', methods=['POST'])
def recommend_behavior_gear():
    data = request.get_json()

    # Extract all behavior-related attributes
    activity_level = data.get("activity_level", "").lower()
    aggression_level = data.get("aggression_level", "").lower()
    climate = data.get("climate", "").lower()

    if not activity_level or not aggression_level or not climate:
        return jsonify({
            "status": "error",
            "message": "Missing required behavior inputs"
        }), 400

    gear = behavior_based_recommendation(activity_level, aggression_level, climate)

    return jsonify({
        "status": "success",
        "recommendations": gear,
        "input": data
    })                         


# Here is a route to take care of image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'dog_image' not in request.files:
        return jsonify({"status": "error", "message": "No file apart"}), 400

    file = request.files['dog_image']

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        # Good practice to secure file name
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file to uploads folder
        file.save(filepath)

        
        # Breed prediction
        try:
            breed_predictions = predict_breed(filepath)
            top_breed = breed_predictions[0]["breed"] if breed_predictions else "Unknown"
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        
        return jsonify({
            "status": "success",
            "message": "Image uploaded",
            "file_path": filepath,
            "breed": top_breed,
            "breed_predictions": breed_predictions
        })  


@app.route('/full_recommendation', methods=['POST'])
def full_recommendation():
    # Extract image and form values
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    
    # We secure filename and save image to our uploads folder
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    # Breed predictor determines top 3 breeds
    try:
        breed_predictions = predict_breed(image_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Breed predcition failed : {str(e)}'}), 500    

    # Extract behavior inputs
    activity_level = request.form.get('activity_level', '').lower()
    aggression_level = request.form.get('aggreesion_level', '').lower()
    climate = request.form.get('climate', '').lower()

    # Recommend gear from behavior
    gear = behavior_based_recommendation(activity_level, aggression_level, climate)

    return jsonify({
        'status': 'success',
        'top_breeds': breed_predictions,
        'gear_recommendation': gear
    })


def behavior_based_recommendation(activity_level, aggression_level, climate):
    collar = "standard padded collar"
    harness = "basic walking harness"
    leash = "standard leash"

    # Aggression logic
    if aggression_level == "high":
        collar = "martingale collar"
        leash = "double-handle leash for control"
    elif activity_level == "moderate":
        leash = "reinforced leash"
    
    # Activity level logic
    if activity_level == "high":
        harness = "Y-front harness with chest padding"
        leash = "shock-absorbing bungee leash"
    elif activity_level == "low":
        harness = "relaxed-fit harness"

    # Climate logic
    if climate == "hot":
        harness = "breathable mesh harness"
    elif climate == "cold":
        harness = "padded insulated harness"

    return {
        "collar": collar,
        "harness": harness,
        "leash": leash
    }



# fallback for undefined pages
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404

if __name__ == '__main__':
    app.run(debug=True)
