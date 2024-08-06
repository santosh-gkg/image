import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load a pretrained YOLO model
model = YOLO("yolov8x.pt")

def extract_text(extracted_objects, output_dir="extracted_objects"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extracted_text = []
    for obj in extracted_objects:
        # Load the extracted object image

        results = model(obj["path"])[0]
        

        objects_text=''
        for box in results.boxes:
            # Get the class ID
            class_id = box.cls.item()
            
            # Get the confidence score
            confidence = box.conf.item()
            
            # Get the class name
            class_name = results.names[int(class_id)]
            objects_text+=f"Object: {class_name}, Confidence: {confidence:.2f}\n"

        print(objects_text)
        extracted_text.append({
            "id": obj["id"],
            "text": objects_text,
            "saved_path": obj["path"],
        })
    
    return extracted_text