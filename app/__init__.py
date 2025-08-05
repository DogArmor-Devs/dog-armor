import os
import logging
import pandas as pd
from flask import Flask

# Create Flask app
app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)

# Configure where uploaded images go
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging
logging.basicConfig(filename='gear_requests.log', level=logging.INFO)

# Load gear recommendation CSV with fallback
try:
    app.gear_data = pd.read_csv('gear_data.csv')
except FileNotFoundError:
    # Create fallback gear data if CSV doesn't exist
    app.gear_data = pd.DataFrame({
        'breed': ['labrador', 'beagle', 'bulldog', 'poodle'],
        'size': ['medium', 'small', 'medium', 'small'],
        'pulls': ['yes', 'no', 'yes', 'no'],
        'budget': ['low', 'medium', 'high', 'medium'],
        'harness_type': ['Y-front', 'step-in', 'vest', 'mesh'],
        'collar_type': ['standard', 'martingale', 'choke', 'flat'],
        'leash_type': ['standard', 'retractable', 'training', 'bungee']
    })

# Define breed labels (fallback list)
app.breed_labels = [
    'labrador', 'beagle', 'bulldog', 'poodle', 'golden_retriever',
    'german_shepherd', 'rottweiler', 'boxer', 'doberman', 'siberian_husky'
]

# Try to load ML model, but provide fallback
app.model = None
app.device = None

try:
    import torch
    from torchvision import models
    
    MODEL_PATH = 'models/retrained_models/breed_classifier.pth'
    if os.path.exists(MODEL_PATH):
        app.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(app.breed_labels))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=app.device))
        model.to(app.device)
        model.eval()
        app.model = model
        print("✅ ML model loaded successfully")
    else:
        print("⚠️  ML model not found, using fallback predictions")
except ImportError:
    print("⚠️  PyTorch not available, using fallback predictions")
except Exception as e:
    print(f"⚠️  Error loading ML model: {e}, using fallback predictions")

# Import routes (must be last so app is defined first)
from app import routes
