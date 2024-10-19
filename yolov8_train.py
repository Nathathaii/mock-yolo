from ultralytics import YOLO
import os
import shutil
import random

model = YOLO("yolov8s.pt")

results = model.train(
    data="yolov8.yaml",
    epochs=2,
    batch=8,
    augment=True,
    # pretrained=False,
    name="yolov8s-epoch5",
)

# Evaluate the model's performance on the validation set
metric = model.val()
# Export the model to ONNX format
success = model.export(format="onnx")