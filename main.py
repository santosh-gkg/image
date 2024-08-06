import os
from segmentation import segment_image
from extract import extract_objects
from identify import identify_objects
from extract_text import extract_text
from summarize import summarize_attributes
from map_data import map_data
from generate_output import generate_output

def main(image_path):
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Image Segmentation
    segmented_image, segmented_objects = segment_image(image_path)
    
    # Step 2: Object Extraction and Storage
    extracted_objects = extract_objects(segmented_image, segmented_objects)
    print(extracted_objects)

    
    # Step 4: Text/Data Extraction from Objects
    extracted_text = extract_text(extracted_objects, output_dir=os.path.join(output_dir, "extracted_objects"))
    
    # Step 5: Summarize Object Attributes
    summarized_attributes = summarize_attributes(extracted_objects, extracted_text)
    
    # Step 6: Data Mapping
    mapped_data = map_data(extracted_objects, extracted_text, summarized_attributes)
    
    # Step 7: Output Generation
    generate_output(image_path, mapped_data)

if __name__ == "__main__":
    image_path = "animals.jpg"
    main(image_path)