import os
import logging
import pandas as pd
from flask import Flask
from src.features.breed_predictor import BREED_LABELS

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

# Load gear recommendation CSV
app.gear_data = pd.read_csv('gear_data.csv')

# Load model
MODEL_PATH = 'models/retrained_models/breed_classifier.pth'
app.model = None
app.device = None
app.breed_labels = BREED_LABELS

try:
    import torch
    from torchvision import models

    if os.path.exists[MODEL_PATH]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(BREED_LABELS))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        app.device = device
        app.model = model
        app.logger.info("Model loaded successfully!")
    else:
        app.logger.warning(f"Model file not found at {MODEL_PATH}. Using fallback predictions.")
except Exception as e:
    app.logger.error(f"Error training model: {e}. Using fallback predictions.")

# Import routes (must be last so app is defined first)
from app import routes