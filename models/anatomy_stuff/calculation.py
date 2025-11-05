import torch
import deeplabcut
from deeplabcut.pose_estimation_pytorch.models import get_model
from deeplabcut.utils.auxiliaryfunctions import read_config
import yaml

# --- 1. Define Paths ---
config_path = r'/path/to/your/project/config.yaml'
# Find your BEST snapshot.pt file (this is the PyTorch model)
snapshot_path = r'/path/to/your/project/dlc-models/iteration-X/shuffle-X/train/snapshot-best.pt'
onnx_output_path = r'./dog_tracker.onnx' # This is your final exported model

# --- 2. Load Model Configuration ---
cfg = read_config(config_path)
num_keypoints = len(cfg['bodyparts'])

# --- 3. Initialize Model Architecture ---
# This MUST match your config.yaml 'net_type' (e.g., 'resnet_50')
model = get_model(cfg['net_type'], num_keypoints=num_keypoints)

# --- 4. Load Your Trained Weights ---
checkpoint = torch.load(snapshot_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set to evaluation mode

# --- 5. Create a Wrapper (CRITICAL for Pytorch DLC Models) ---
# DLC models output a complex dictionary. ONNX needs a simple
# tensor output. This wrapper flattens it.
class ExportableModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # We only care about the heatmap for keypoints
        outputs = self.model(x)
        return outputs['heatmap'] # Or outputs[0] for older DLC versions

export_model = ExportableModel(model)

# --- 6. Define Dummy Input ---
# The input size MUST match your model's training size
# Check your config.yaml for 'crop_size' or 'image_dimensions'
# Common sizes are (256, 256), (400, 400), etc.
INPUT_H = 400 
INPUT_W = 400
dummy_input = torch.randn(1, 3, INPUT_H, INPUT_W)

# --- 7. Export! ---
print(f"Exporting model to {onnx_output_path}...")
torch.onnx.export(
    export_model,
    dummy_input,
    onnx_output_path,
    input_names=['input'],
    output_names=['heatmap'],
    opset_version=12,
    export_params=True
)
print("Export complete!")
