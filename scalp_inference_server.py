from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import requests

app = Flask(__name__)

# S3에서 다운로드할 모델 URL 목록
model_urls = {
    "미세각질": "https://smartbrush-models.s3.ap-northeast-2.amazonaws.com/model1_full.pt",
    "탈모": "https://smartbrush-models.s3.ap-northeast-2.amazonaws.com/model2_full.pt",
    "모낭사이홍반": "https://smartbrush-models.s3.ap-northeast-2.amazonaws.com/model3_full.pt",
    "모낭홍반농포": "https://smartbrush-models.s3.ap-northeast-2.amazonaws.com/model4_full.pt",
    "비듬": "https://smartbrush-models.s3.ap-northeast-2.amazonaws.com/model5_full.pt",
    "피지과다": "https://smartbrush-models.s3.ap-northeast-2.amazonaws.com/model6_full.pt",
}

# 모델 캐시 딕셔너리
loaded_models = {}

# 모델을 /tmp 에 다운로드 후 로드
def get_model(disease):
    if disease in loaded_models:
        return loaded_models[disease]

    url = model_urls[disease]
    local_path = f"/tmp/{disease}.pt"

    # 모델 파일 없으면 다운로드
    if not os.path.exists(local_path):
        print(f"Downloading model for {disease}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

    # 모델 로딩
    model = torch.load(local_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    loaded_models[disease] = model
    return model

# 이미지 전처리
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
    for disease in model_urls.keys():
        model = get_model(disease)
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
