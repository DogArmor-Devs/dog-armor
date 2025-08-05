# New file for predictions, connect to Flask route

import torch
from torchvision import transforms, models
from PIL import Image
import pickle
import os

BREED_LABELS = [
    "Chihuahua",
    "Japanese_spaniel",
    "Maltese_dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim_spaniel",
    "papillon",
    "toy_terrier",
    "Rhodesian_ridgeback",
    "Afghan_hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan_coonhound",
    "Walker_hound",
    "English_foxhound",
    "redbone",
    "borzoi",
    "Irish_wolfhound",
    "Italian_greyhound",
    "whippet",
    "Ibizan_hound",
    "Norwegian_elkhound",
    "otterhound",
    "Saluki",
    "Scottish_deerhound",
    "Weimaraner",
    "Staffordshire_bullterrier",
    "American_Staffordshire_terrier",
    "Bedlington_terrier",
    "Border_terrier",
    "Kerry_blue_terrier",
    "Irish_terrier",
    "Norfolk_terrier",
    "Norwich_terrier",
    "Yorkshire_terrier",
    "wire-haired_fox_terrier",
    "Lakeland_terrier",
    "Sealyham_terrier",
    "Airedale",
    "cairn",
    "Australian_terrier",
    "Dandie_Dinmont",
    "Boston_bull",
    "miniature_schnauzer",
    "giant_schnauzer",
    "standard_schnauzer",
    "Scotch_terrier",
    "Tibetan_terrier",
    "silky_terrier",
    "soft-coated_wheaten_terrier",
    "West_Highland_white_terrier",
    "Lhasa",
    "flat-coated_retriever",
    "curly-coated_retriever",
    "golden_retriever",
    "Labrador_retriever",
    "Chesapeake_Bay_retriever",
    "German_short-haired_pointer",
    "vizsla",
    "English_setter",
    "Irish_setter",
    "Gordon_setter",
    "Brittany_spaniel",
    "clumber",
    "English_springer",
    "Welsh_springer_spaniel",
    "cocker_spaniel",
    "Sussex_spaniel",
    "Irish_water_spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old_English_sheepdog",
    "Shetland_sheepdog",
    "collie",
    "Border_collie",
    "Bouvier_des_Flandres",
    "Rottweiler",
    "German_shepherd",
    "Doberman",
    "miniature_pinscher",
    "Greater_Swiss_Mountain_dog",
    "Bernese_mountain_dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull_mastiff",
    "Tibetan_mastiff",
    "French_bulldog",
    "Great_Dane",
    "Saint_Bernard",
    "Eskimo_dog",
    "malamute",
    "Siberian_husky",
    "affenpinscher",
    "basenji",
    "pug",
    "Leonberg",
    "Newfoundland",
    "Great_Pyrenees",
    "Samoyed",
    "Pomeranian",
    "chow",
    "keeshond",
    "Brabancon_griffon",
    "Pembroke",
    "Cardigan",
    "toy_poodle",
    "miniature_poodle",
    "standard_poodle",
    "Mexican_hairless",
    "dingo",
    "dhole",
    "African_hunting_dog"
]

# Paths
MODEL_PATH = os.path.join("src", "models", "dog_breed_model.pth")
ENCODER_PATH = os.path.join("src", "models", "label_encoder.pkl")

# Load label encoder
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Predict function
def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return class_name