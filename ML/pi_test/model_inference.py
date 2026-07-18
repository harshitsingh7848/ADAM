import os
import time
import torch
from torchvision import transforms
from PIL import Image

# Define the path to the model and images directory
MODEL_PATH = "trained_model.pt"
IMAGES_DIR = "images"

def load_model():
    """Load the trained PyTorch model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    model = torch.load(MODEL_PATH)
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(image_path):
    """Preprocess the image to be suitable for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust based on your model's requirements
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def run_inference(model, image_tensor):
    """Run inference on a single image tensor."""
    with torch.no_grad():
        start_time = time.time()
        output = model(image_tensor)
        inference_time = time.time() - start_time
    return output, inference_time

def main():
    """Main function to perform inference on all images."""
    # Load the model
    model = load_model()

    # Check if the images directory exists
    if not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError(f"Images directory '{IMAGES_DIR}' not found.")

    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print("No images found in the directory.")
        return

    print(f"Running inference on {len(image_files)} images...")

    for image_file in image_files:
        image_path = os.path.join(IMAGES_DIR, image_file)

        try:
            # Preprocess the image
            image_tensor = preprocess_image(image_path)

            # Run inference
            output, inference_time = run_inference(model, image_tensor)

            # Print the result
            print(f"Image: {image_file} | Inference Time: {inference_time:.4f} seconds")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

if __name__ == "__main__":
    main()
