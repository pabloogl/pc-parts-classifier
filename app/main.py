import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="PC Parts Classifier")

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# Load labels
LABELS_PATH = BASE_DIR / "models" / "labels.json"
labels = json.loads(LABELS_PATH.read_text())

# Load model
def load_model():
    model_path = BASE_DIR / "models" / "model.pt"
    checkpoint = torch.load(model_path, map_location="cpu")
    arch = checkpoint.get("arch", "resnet18")
    
    if arch == "resnet18":
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(labels))
    else:
        # Fallback/Default for efficiency (could be expanded)
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(labels))
        
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    confidences = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    
    top_prediction = sorted_confidences[0]
    
    return {
        "prediction": top_prediction[0],
        "confidence": top_prediction[1],
        "all_predictions": sorted_confidences[:5]
    }
