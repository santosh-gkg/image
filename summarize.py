from transformers import pipeline
from ultralytics import YOLO
import cv2
import numpy as np

summarizer = pipeline("summarization")
yolo_model = YOLO("yolov8x.pt")

def get_position(x, y):
    if x < 0.33:
        h_pos = "left"
    elif x < 0.66:
        h_pos = "center"
    else:
        h_pos = "right"
    
    if y < 0.33:
        v_pos = "top"
    elif y < 0.66:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    return f"{v_pos}-{h_pos}"

def get_size(width, height):
    area = width * height
    if area < 0.1:
        return "small"
    elif area < 0.3:
        return "medium"
    else:
        return "large"

def summarize_object_attributes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Failed to load image"

    results = yolo_model(image)
    height, width = image.shape[:2]
    
    summaries = []
    for result in results:
        for detection in result.boxes.data:
            class_id = int(detection[5])
            object_name = result.names[class_id]
            confidence = detection[4].item()
            x1, y1, x2, y2 = detection[:4].tolist()
            
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            position = get_position(center_x, center_y)
            
            obj_width = (x2 - x1) / width
            obj_height = (y2 - y1) / height
            size = get_size(obj_width, obj_height)
            
            summary = f"Object: {object_name}, Confidence: {confidence:.2f}, Position: {position}, Size: {size}"
            summaries.append(summary)
    
    return ". ".join(summaries)

def summarize_attributes(extracted_objects,extracted_text):
    summarized_attributes = []
    for obj, text in zip(extracted_objects,extracted_text):
        image_path = obj["path"]
        category = text["text"]
        
        
        object_summary = summarize_object_attributes(image_path)
        
        combined_text = f"{category}. {object_summary}"
        
        summary=combined_text
        
        summarized_attributes.append({
            "id": obj["id"],
            "category": category,
            "summary": summary
        })
    
    return summarized_attributes