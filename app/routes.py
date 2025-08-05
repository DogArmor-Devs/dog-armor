from flask import request, jsonify, render_template
from app import app
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import random

# Simple pages
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


# 🧠 Gear recommendation
@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()

    breed = data.get("breed", "").lower()
    size = data.get("size", "").lower()
    puller = data.get("puller", "").lower()
    budget = data.get("budget", "").lower()

    gear_data = app.gear_data

    # Basic filtering
    filtered = gear_data[
        (gear_data["breed"].str.lower().str.contains(breed, na=False)) |
        (gear_data["size"].str.lower().str.contains(size, na=False)) |
        (gear_data["pulls"].str.lower().str.contains(puller, na=False)) |
        (gear_data["budget"].str.lower().str.contains(budget, na=False))
    ]

    if filtered.empty:
        recommendation = gear_data.sample(1).to_dict(orient="records")[0]
    else:
        recommendation = filtered.sample(1).to_dict(orient="records")[0]

    # Log
    app.logger.info(f"[{datetime.now()}] /recommend requested with data: {data}")

    return jsonify({
        "recommendations": recommendation,
        "input": data
    })


# 🖼️ Image upload and breed prediction with fallback
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'dog_image' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['dog_image']

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict breed with fallback
        try:
            if app.model is not None:
                # Use ML model if available
                from PIL import Image
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                
                image = Image.open(filepath).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(app.device)
                
                with torch.no_grad():
                    outputs = app.model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_breed = app.breed_labels[predicted.item()]
            else:
                # Fallback: random breed prediction
                predicted_breed = random.choice(app.breed_labels)
                
        except Exception as e:
            # Ultimate fallback
            predicted_breed = random.choice(app.breed_labels)

        return jsonify({
            "status": "success",
            "message": "Image uploaded",
            "file_path": filepath,
            "breed": predicted_breed
        })


# Full recommendation pipeline
@app.route('/full_recommendation', methods=['POST'])
def full_recommendation():
    # Extract image and form values
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    
    # Secure filename and save image
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    # Breed prediction with fallback
    try:
        if app.model is not None:
            from PIL import Image
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(app.device)
            
            with torch.no_grad():
                outputs = app.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_breed = app.breed_labels[predicted.item()]
        else:
            predicted_breed = random.choice(app.breed_labels)
    except Exception as e:
        predicted_breed = random.choice(app.breed_labels)

    # Extract behavior inputs
    activity_level = request.form.get('activity_level', '').lower()
    aggression_level = request.form.get('aggression_level', '').lower()
    climate = request.form.get('climate', '').lower()

    # Generate gear recommendations
    gear = behavior_based_recommendation(activity_level, aggression_level, climate)

    return jsonify({
        'status': 'success',
        'predicted_breed': predicted_breed,
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
    elif aggression_level == "moderate":
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


# 404 page
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404
