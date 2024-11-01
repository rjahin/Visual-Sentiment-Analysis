import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


categories = ["Person/People", "Sports", "Clothes", "Tattoos", "Animal"]
emotions = ["Angry", "Sad", "Happy", "Sarcastic", "Mocking"]
text_in_image = ["Text", "Overlaid Text", "an image without text"]
targets = ["Person/People", "Celebrity", "Famous personality"]
bullying_phrases = ["Loser", "Ugly", "Embarrassing", "Stupid", "Fat"]


categories_tokens = clip.tokenize(categories).to(device)
emotions_tokens = clip.tokenize(emotions).to(device)
text_in_image_tokens = clip.tokenize(text_in_image).to(device)
targets_tokens = clip.tokenize(targets).to(device)
bullying_phrases_tokens = clip.tokenize(bullying_phrases).to(device)

def analyze_image(image_path):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image)


        categories_features = model.encode_text(categories_tokens)
        category_similarity = (image_features @ categories_features.T).softmax(dim=-1)
        print("Contextual Categories :")
        for i, category in enumerate(categories):
            print(f"{category}: {category_similarity[0, i].item():.2f}")


        emotions_features = model.encode_text(emotions_tokens)
        emotion_similarity = (image_features @ emotions_features.T).softmax(dim=-1)
        print("\nEmotions :")
        for i, emotion in enumerate(emotions):
            print(f"{emotion}: {emotion_similarity[0, i].item():.2f}")


        text_in_image_features = model.encode_text(text_in_image_tokens)
        text_in_image_similarity = (image_features @ text_in_image_features.T).softmax(dim=-1)
        print("\nText Presence :")
        for i, text_presence in enumerate(text_in_image):
            print(f"{text_presence}: {text_in_image_similarity[0, i].item():.2f}")


        targets_features = model.encode_text(targets_tokens)
        target_similarity = (image_features @ targets_features.T).softmax(dim=-1)
        print("\nPotential Targets :")
        for i, target in enumerate(targets):
            print(f"{target}: {target_similarity[0, i].item():.2f}")


        phrases_features = model.encode_text(bullying_phrases_tokens)
        bullying_similarity = (image_features @ phrases_features.T).softmax(dim=-1)
        print("\nBullying Phrases :")
        for i, phrase in enumerate(bullying_phrases):
            print(f"{phrase}: {bullying_similarity[0, i].item():.2f}")


image_path = "woman.jpeg"
analyze_image(image_path)
