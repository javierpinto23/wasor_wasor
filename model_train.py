from ultralytics import YOLO
import torch

# Load the model
model = YOLO("model/wasor_model.pt")

# Train the model
results = model.train(data="/Users/pintojav/Desktop/wasor_wasor/wasor_wasor/wasor_data.yaml", epochs=2, imgsz=640, device="mps", project="wasor_wasor",name="runs")