import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

def generate_output(image_path, mapped_data):
    # Load the original image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw bounding boxes and labels
    for obj in mapped_data:
        box = obj["box"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"ID: {obj['id']}", fill="red")
    
    # Save the annotated image
    image.save("output_image.jpg")
    
    # Create a summary table
    df = pd.DataFrame(mapped_data)
    df.to_csv("output_summary.csv", index=False)
    
    # Display the image and table
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Annotated Image")
    plt.show()
    
    print(df)
    