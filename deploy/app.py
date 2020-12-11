from flask import Flask, jsonify, request, render_template
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from PIL import Image
from PIL import Image
import numpy as np
import torch
import io, re, os
from processing import answerGet


app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()


def transform_image(ans):
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    tensors = torch.stack([trans(Image.fromarray(item)) for item in ans], 0)
    return tensors


def get_prediction(ans):
    tensor = transform_image(ans)
    class_names = ['A','B','C','D']
    with torch.no_grad():
        outputs = model.forward(tensor)
        preds = outputs.argmax(1)
        prediction = [class_names[item] for item in preds]
        confidence = F.softmax(outputs, 1).max(1)[0].detach().numpy().tolist()
        return prediction, confidence


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('L')
    img = np.array(img)
    part = request.values.get( "part" )
    ans = answerGet(img, int(part))
    prediction, confidence = get_prediction(ans)
    return jsonify({'prediction': prediction, 'confidence': confidence})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5000')
