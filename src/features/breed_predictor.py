# New file for predictions, connect to Flask route

import torch
from torchvision import transforms, models
from PIL import Image
import joblib

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
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3 = torch.topk(probabilities, 3)

    top3_indices = top3.indices.tolist()
    top3_scores = top3.values.tolist()
    top3_breeds = encoder.inverse_transform(top3_indices)

    results = []
    for breed, score in zip(top3_breeds, top3_scores):
        results.append({
            "breed": breed,
            "confidence": round(float(score), 4)    # convert to float and round
        })

    return results
    