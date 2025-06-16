# New file for predictions, connect to Flask route

import torch
from torchvision import transforms, models
from PIL import Image
import joblib

device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")

# Load model and label encoder
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 120)  # 120 breeds
model.load_state_dict(torch.load("model/dog_breed_model.pth", map_location = device))
model.eval()

encoder = joblib.load("model/label_encoder.pkl")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_breed(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()
        breed = encoder.inverse_transform([predicted_class])[0]
        return breed