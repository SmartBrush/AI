from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load all models
model_paths = {
    "미세각질": "model1_full.pt",
    # "탈모": "model2_full.pt",
    # "모낭사이홍반": "model3_full.pt",
    "모낭홍반농포": "model4_full.pt",
    "비듬": "model5_full.pt",
    # "피지과다": "model6_full.pt"
}

models = {}
for name, path in model_paths.items():
    models[name] = torch.load(path, map_location=torch.device('cpu'))
    models[name].eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize([600, 600]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/ai", methods=["POST"])
def ai():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = Image.open(request.files['image']).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    results = {}
    for disease, model in models.items():
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = torch.argmax(prob).item()
            results[disease] = {
                "class_index": pred_class,
                "confidence": round(prob[pred_class].item(), 3)
            }

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000)
