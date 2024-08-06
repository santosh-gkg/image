import numpy as np
from PIL import Image
import os



# Add this at the beginning of your main function
if not os.path.exists("extracted_objects"):
    os.makedirs("extracted_objects")


def extract_objects(image, segmented_objects):
    extracted_objects = []
    for i, (mask, box) in enumerate(segmented_objects):
        print(f"Image shape: {np.array(image).shape}, dtype: {np.array(image).dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        # Ensure mask is 2D and boolean
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        mask = mask.astype(bool)

        # Convert image to numpy array if it's not already
        image_array = np.array(image)

        # Apply mask to original image
        masked_image = image_array * mask[:, :, np.newaxis]
        
        # Crop the object using the bounding box
        x1, y1, x2, y2 = map(int, box)
        cropped_object = masked_image[y1:y2, x1:x2]

        # Ensure the cropped object is not empty
        if cropped_object.size > 0:
            # Convert to uint8 and create PIL Image
            cropped_object = (cropped_object * 255).astype(np.uint8)
            cropped_image = Image.fromarray(cropped_object)
            
            # Save the cropped object

            object_path = f"extracted_objects/object_{i}.png"
            cropped_image.save(object_path)
            
            extracted_objects.append({
                "id": i,
                "path": object_path,
                "box": box
            })
    print(extracted_objects)
    
    return extracted_objects