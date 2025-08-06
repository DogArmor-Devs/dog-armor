import os
import logging
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
from torchvision import transforms
from src.features.breed_predictor import predict_breed

main = Blueprint("main", __name__)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -- PAGE ROUTES --

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/demo')
def demo():
    return render_template('demo.html')

@main.route('/features')
def features():
    return render_template('features.html')

@main.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@main.route('/why-us')
def why_us():
    return render_template('why-us.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/team')
def team():
    return render_template('team.html')

# -- IMAGE UPLOAD --

@main.route('/upload', methods=['POST'])
def upload_image():
    if 'dog_image' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files['dog_image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

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

# -- FORM & BEHAVIOR ROUTES --

@main.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()
    gear_data = current_app.gear_data

    breed = data.get("breed", "").lower()
    weight = data.get("weight", "").lower()
    neck = data.get("neck_circumference", "").lower()
    pull = data.get("pull", "").lower()
    budget = data.get("budget", "").lower()

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

    logging.info(f"[{datetime.now()}] /recommend requested with data: {data}")

    return jsonify({"recommendations": recommendation, "input": data})

@main.route('/behavior-gear', methods=['POST'])
def recommend_behavior_gear():
    data = request.get_json()
    activity = data.get("activity_level", "").lower()
    aggression = data.get("aggression_level", "").lower()
    climate = data.get("climate", "").lower()

    if not activity or not aggression or not climate:
        return jsonify({"status": "error", "message": "Missing behavior inputs"}), 400

    gear = behavior_based_recommendation(activity, aggression, climate)

    return jsonify({"status": "success", "recommendations": gear, "input": data})

@main.route('/full_recommendation', methods=['POST'])
def full_recommendation():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    try:
        breed_predictions = predict_breed(image_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Breed prediction failed: {str(e)}'}), 500

    activity = request.form.get('activity_level', '').lower()
    aggression = request.form.get('aggression_level', '').lower()
    climate = request.form.get('climate', '').lower()
    water_activities = request.form.get('water_activities_yn', '').lower()
    heat_sensitive = request.form.get('heat_sensitive_yn', '').lower()

    material_note = get_material_note(activity, water_activities, heat_sensitive)
    gear = behavior_based_recommendation(activity, aggression, climate)
    gear['material_note'] = material_note

    return jsonify({
        'status': 'success',
        'top_breeds': breed_predictions,
        'gear_recommendation': gear
    })

# -- SUPPORT FUNCTIONS --

def get_material_note(activity, water, heat_sensitive):
    if heat_sensitive == "yes":
        return "Mesh: Lightweight and breathable, ideal for hot weather or dogs prone to overheating."
    if water == 'yes':
        return "Neoprene: Cushioned and sweet for sensitive skin and water activities."
    if activity == 'high':
        return "Nylon: Durable, lightweight, and great for active dogs."
    return "Nylon: Durable and affordable."

def behavior_based_recommendation(activity, aggression, climate):
    collar = "standard padded collar"
    harness = "basic walking harness"
    leash = "standard leash"

    if aggression == "high":
        collar = "martingale collar"
        leash = "double-handle leash"
    elif activity == "moderate":
        leash = "reinforced leash"

    if activity == "high":
        harness = "Y-front harness with chest padding"
        leash = "shock-absorbing bungee leash"
    elif activity == "low":
        harness = "relaxed-fit harness"

    if climate == "hot":
        harness = "breathable mesh harness"
    elif climate == "cold":
        harness = "padded insulated harness"

    return {"collar": collar, "harness": harness, "leash": leash}

# -- ERROR HANDLER --

@main.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404
