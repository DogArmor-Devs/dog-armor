import os
import logging
import torch
import pandas as pd
from flask import Flask
from torchvision import models

# Create Flask app
app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)

# Configure where uploaded images go
UPLOAD_FOLDER = os.path.join(app.static_folder)  # everything in static/
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging
logging.basicConfig(filename='gear_requests.log', level=logging.INFO)

# Load gear recommendation CSV with fallback
try:
    app.gear_data = pd.read_csv('gear_data.csv')
except FileNotFoundError:
    # Create fallback gear data
    app.gear_data = pd.DataFrame({
        'breed': ['labrador', 'beagle', 'bulldog', 'poodle'],
        'size': ['large', 'medium', 'large', 'medium'],
        'pulls': ['yes', 'no', 'yes', 'no'],
        'budget': ['medium', 'low', 'high', 'medium'],
        'gear_type': ['harness', 'collar', 'leash', 'toys'],
        'recommendation': ['Active dog harness', 'Comfortable collar', 'Strong leash', 'Interactive toys']
    })

# Define breed labels (replace with your actual list)
BREED_LABELS = [
    'labrador', 'beagle', 'bulldog', 'poodle'
    # Add your actual classes here
]

# Load model with fallback
MODEL_PATH = 'models/retrained_models/breed_classifier.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(BREED_LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    app.model_loaded = True
except (FileNotFoundError, Exception) as e:
    print(f"Model not found or error loading: {e}")
    model = None
    app.model_loaded = False

# Attach to app for routes to use
app.device = device
app.model = model
app.breed_labels = BREED_LABELS

# Import routes (must be last so app is defined first)
from app import routes
