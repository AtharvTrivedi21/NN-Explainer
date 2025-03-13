import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet, DenseNet121, EfficientNetB0
from tensorflow.keras.utils import plot_model
import os

# Create a folder to store images
output_dir = "model_images"
os.makedirs(output_dir, exist_ok=True)

# Dictionary of models and output file names
models = {
    "VGG16": VGG16(weights=None),
    "ResNet50": ResNet50(weights=None),
    "InceptionV3": InceptionV3(weights=None),
    "MobileNet": MobileNet(weights=None),
    "DenseNet121": DenseNet121(weights=None),
    "EfficientNetB0": EfficientNetB0(weights=None)
}

# Generate and save architecture images
for model_name, model in models.items():
    file_path = os.path.join(output_dir, f"{model_name.lower()}.png")
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True, rankdir="TB")
    print(f"Saved {file_path}")

print("All model diagrams generated successfully!")
