import os
from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class AlzheimerModel(torch.nn.Module):
    def __init__(self):
        super(AlzheimerModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 4)
        )
    
    def forward(self, x):
        return self.base_model(x)

model = AlzheimerModel()
model.load_state_dict(torch.load('alzheimer_model.pth', map_location=torch.device('cpu')))
model.to(torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

labels = ['Mild Impairment', 'Moderate Impairment', 'Non Impairment', 'Very Mild Impairment']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file!"
    
    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = Image.open(filepath).convert('RGB')  
    image = transform(image).unsqueeze(0)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    image = image.to(device)  

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = labels[predicted.item()]
    
    return render_template('result.html', image_url=filepath, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
