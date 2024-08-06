import torch
from torchvision.models import resnet50
from torchvision.transforms import functional as F
from PIL import Image

# Load pre-trained ResNet model
model = resnet50(pretrained=True)
model.eval()



def identify_objects(extracted_objects):
    identified_objects = []
    for obj in extracted_objects:
        image = Image.open(obj["path"])
        image_tensor = F.to_tensor(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
        
        _, predicted_idx = torch.max(output, 1)
        category = categories[predicted_idx.item()]
        
        identified_objects.append({
            "id": obj["id"],
            "category": category
        })
    
    return identified_objects