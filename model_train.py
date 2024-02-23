from ultralytics import YOLO
import os

# Load the model
model = YOLO("model/wasor_model.pt")

# Train the model
results = model.train(data="/Users/pintojav/Desktop/wasor_wasor/wasor_wasor/wasor_data.yaml", epochs=1, imgsz=640, device='cpu', project="wasor_wasor",name="runs")