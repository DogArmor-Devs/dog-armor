"""Export the breed classifier to ONNX for production inference.

The web app serves predictions with onnxruntime only (no PyTorch at runtime).
This script is a one-off build step; it needs torch + torchvision + onnx:

    pip install torch torchvision onnx onnxscript
    python -m src.features.export_onnx

The model is torchvision's EfficientNetV2-S with ImageNet-1k weights. ImageNet
already contains every Stanford Dogs breed as a class (indices 151-268), and on
held-out photos it beat our previous fine-tuned ResNet-18 by a wide margin
(81% vs 34% top-1 on real-world photos), while its non-dog classes give us a
free "is there a dog in this photo?" check. The exported graph returns softmax
probabilities over all 1000 classes; data/processed_data/breed_profiles.csv
maps the dog class indices to breed names and body measurements.
"""

import argparse
import os

import torch
from torchvision import models

DEFAULT_OUTPUT = os.path.join("models", "breed_classifier.onnx")
INPUT_SIZE = 384


class WithSoftmax(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).eval()
    torch.onnx.export(
        WithSoftmax(model),
        torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),
        args.output,
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported EfficientNetV2-S -> {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
