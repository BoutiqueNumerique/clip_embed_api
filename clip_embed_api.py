from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import snapshot_download
from PIL import Image
import torch
import io
import os

app = Flask(__name__)

# 📦 Télécharger le modèle dans un dossier temporaire local (évite les erreurs de cache)
MODEL_DIR = "/tmp/clip-model"
snapshot_download("openai/clip-vit-base-patch32", local_dir=MODEL_DIR, local_dir_use_symlinks=False)

# 🧠 Charger le modèle et le processor
model = CLIPModel.from_pretrained(MODEL_DIR)
processor = CLIPProcessor.from_pretrained(MODEL_DIR)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
