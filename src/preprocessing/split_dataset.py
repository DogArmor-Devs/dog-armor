# Convert breed names to numerical class labels
# Split into training & validation sets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

df = pd.read_csv('data/dogarmor_dataset.csv')

# Encode breed names to integers
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['breed'])

# Save the encoder for use in inference
import joblib
os.makedirs('model', exist_ok=True)
joblib.dump(encoder, 'model/label_encoder.pkl')

# Here we split into training & testing
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)

print(f" Train size: {len(train_df)}, Validation size: {len(val_df)}")