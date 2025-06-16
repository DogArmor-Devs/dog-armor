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
from utils.breed_predictor import predict_breed

app = Flask(__name__, static_folder='static', template_folder='templates')

# A folder inside static called 'uploads' for storing 
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Breed classifier model
MODEL_PATH = 'model/breed_classifier.pth'

# Class labels
BREED_LABELS = [
    'beagle', 'chihuahua', 'golden_retriever', 'pug', 'tibetan_terrier', 'doberman',
    'scottish_deerhound', 'malamute', 'saint_bernard'
    # Add more..
]

# For loading models and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(BREED_LABELS))
model.load_state_dict(torch.load(MODEL_PATH), map_location = device)
model.to(device)
model.eval()

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

    # Get user answers from the demo form   
    breed = data.get("breed", "").lower()
    size = data.get("size", "").lower()
    puller = data.get("puller", "").lower()
    budget = data.get("budget", "").lower()

   # 🧠 Very basic filtering: check if any matching rows exist
    filtered = gear_data[
        (gear_data["breed"].str.lower().str.contains(breed.lower(), na=False)) |
        (gear_data["size"].str.lower().str.contains(size.lower(), na=False)) |
        (gear_data["pulls"].str.lower().str.contains(puller.lower(), na=False)) |
        (gear_data["budget"].str.lower().str.contains(budget.lower(), na=False))   
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
            predicted_breed = predict_breed(filepath)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        
        return jsonify({
            "status": "success",
             "message": "Image uploaded",
             "file_path": filepath,
             "breed": predicted_breed
            })    


# fallback for undefined pages
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404

if __name__ == '__main__':
    app.run(debug=True)
