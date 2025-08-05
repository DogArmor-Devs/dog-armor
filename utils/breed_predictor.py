import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random

def predict_breed(image_path, model=None, device=None, labels=None):
    """
    Predict dog breed from image.
    
    Args:
        image_path (str): Path to the image file
        model: PyTorch model (can be None for fallback)
        device: Device to run inference on
        labels (list): List of breed labels
    
    Returns:
        str: Predicted breed
    """
    
    # If no model is available, use fallback
    if model is None or labels is None:
        return fallback_breed_prediction()
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            
        return labels[predicted_idx]
        
    except Exception as e:
        print(f"Error in breed prediction: {e}")
        return fallback_breed_prediction()

def fallback_breed_prediction():
    """
    Fallback breed prediction when model is not available.
    Returns a random breed from common breeds.
    """
    common_breeds = [
        'labrador', 'beagle', 'bulldog', 'poodle', 'golden_retriever',
        'german_shepherd', 'rottweiler', 'yorkshire_terrier', 'boxer',
        'dachshund', 'chihuahua', 'shih_tzu', 'great_dane', 'siberian_husky'
    ]
    return random.choice(common_breeds)