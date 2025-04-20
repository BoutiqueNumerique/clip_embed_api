from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = Flask(__name__)

# Charger une seule fois le modèle CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

@app.route('/embed', methods=['POST'])
def embed_image():
    if 'image' not in request.files:
        return jsonify({"error": "Image manquante"}), 400

    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            vector = outputs[0].tolist()

        return jsonify({"clip_vector": vector})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
