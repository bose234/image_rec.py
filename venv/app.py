from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import requests

app = Flask(__name__)

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = np.array(img)  # Convert to NumPy array
    img = np.expand_dims(img, axis=0)  # Add a new axis

    # Make the prediction
    with torch.no_grad():
        output = model(torch.tensor(img))

    # Load the labels used by the pre-trained model
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(labels_url)
    labels = response.json()

    # Get the predicted label
    _, predicted_idx = torch.max(output, 1)
    predicted_label = labels[predicted_idx.item()]

    return predicted_label

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_path = 'temp.jpg'
    file.save(image_path)

    try:
        predicted_label = predict_image(image_path)
        return jsonify({'predicted_label': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
