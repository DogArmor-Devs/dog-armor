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

# Load gear recommendation CSV
app.gear_data = pd.read_csv('gear_data.csv')

# Define breed labels (replace with your actual list)
BREED_LABELS = [
    'labrador', 'beagle', 'bulldog', 'poodle'
    # Add your actual classes here
]

# Load model
MODEL_PATH = 'models/retrained_models/breed_classifier.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(BREED_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Attach to app for routes to use
app.device = device
app.model = model
app.breed_labels = BREED_LABELS

# Import routes (must be last so app is defined first)
from app import routes
