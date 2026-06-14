import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import gradio as gr

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model definition
def get_efficientnet_b0(num_classes=7):
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

# Load model
model = get_efficientnet_b0()
model.load_state_dict(torch.load('best_efficientnet_b0.pth', map_location=DEVICE))
model.eval().to(DEVICE)

# Transform
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Predict function
@torch.no_grad()
def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)["image"]
    tensor = transformed.unsqueeze(0).to(DEVICE)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

# Gradio UI
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Face Image"),
    outputs=gr.Label(num_top_classes=7, label="Emotion Probabilities"),
    title="😊 Facial Emotion Detection",
    description="Upload a face image to detect emotion using EfficientNet-B0 trained on FER-2013 (35K+ images) | 7 Emotions | ~73% Accuracy",
    theme=gr.themes.Soft()
)

app.launch()