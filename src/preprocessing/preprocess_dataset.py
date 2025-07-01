# Traverse all subfolders in Images/
# Extract the breed name (e.g., Labrador)
# Save the image path and breed label for training

import os
import pandas as pd

DATA_DIR = 'data/stanford-dogs/images/Images'
records = []

for breed_folder in os.listdir(DATA_DIR):
    breed_path = os.path.join(DATA_DIR, breed_folder)
    if not os.path.isdir(breed_path):
        continue

    # Get breed name from folder (e.g. "n02085620-Chihuahua" → "Chihuahua")
    breed_name = breed_folder.split('-')[-1].replace('_', ' ')

    for img_file in os.listdir(breed_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(breed_path, img_file)
            records.append({'filepath': img_path, 'breed': breed_name})


# Save this data as CSV for easy dataset loading
df = pd.DataFrame(records)
df.to_csv('data/dogarmor_dataset.csv', index=False)

print(f" Saved {len(df)} image entries.")
