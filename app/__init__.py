import os
import logging
import pandas as pd
from flask import Flask
from torchvision.models import resnet50
from src.features.breed_predictor import BREED_LABELS

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')

    # Upload folder setup
    UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Logging
    logging.basicConfig(filename='gear_requests.log', level=logging.INFO)

    # Load gear CSV
    app.gear_data = pd.read_csv('gear_data.csv')

    # Load trained model
    MODEL_PATH = 'models/retrained_models/breed_classifier.pth'
    app.model = None
    app.device = None
    app.breed_labels = BREED_LABELS

    try:
        import torch
        from torchvision import models
        if os.path.exists(MODEL_PATH):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(BREED_LABELS))
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            app.device = device
            app.model = model
            app.logger.info("Model loaded successfully!")
        else:
            app.logger.warning(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")

    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    return app

app = create_app()
