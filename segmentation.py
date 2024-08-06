import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

def segment_image(image_path):
    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # Perform segmentation
    with torch.no_grad():
        prediction = model(image_tensor)

    # Extract masks and boxes
    masks = prediction[0]['masks'].squeeze().cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()

    return image, list(zip(masks, boxes))