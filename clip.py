import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define bullying-related keywords
bullying_keywords = ["ugly", "stupid", "hate", "loser", "worthless", "fail"]

def load_image(image_path):
    """Load an image from the file path."""
    image = Image.open(image_path).convert("RGB")
    return image

def generate_bullying_score(text, image):
    """Evaluate the similarity between text and image with a bullying focus."""
    # Process the text and image
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    
    # Calculate the similarity score between image and text
    logits_per_image = outputs.logits_per_image
    similarity_score = logits_per_image.item()

    # Check if text contains any bullying keywords
    bullying_score = sum(1 for word in bullying_keywords if word in text.lower())

    # Adjust final score: higher scores if keywords are present
    final_score = similarity_score + bullying_score * 0.5  # weight for bullying words

    return final_score

# Example usage
image_path = "path/to/instagram_image.jpg"  # Replace with the actual path
text = "You're so ugly and dumb!"  # Example bullying text

image = load_image(image_path)
bullying_score = generate_bullying_score(text, image)

print(f"Bullying score for this post: {bullying_score}")
