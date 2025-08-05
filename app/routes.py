from flask import request, jsonify, render_template
from app import app
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from PIL import Image
from src.features.breed_predictor import predict_breed

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


# 🖼️ Image upload and breed prediction
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

        # Predict
        try:
            predicted_breed = predict_breed(
                filepath,
                model=app.model,
                device=app.device,
                labels=app.breed_labels
            )
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        return jsonify({
            "status": "success",
            "message": "Image uploaded",
            "file_path": filepath,
            "breed": predicted_breed
        })


# 404 page
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404
