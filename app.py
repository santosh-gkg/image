import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd

from segmentation import segment_image
from extract import extract_objects
from identify import identify_objects
from extract_text import extract_text
from summarize import summarize_attributes
from map_data import map_data



def process_image(image):
    # Save the uploaded image temporarily
    temp_image_path = "sample.jpg"
    output_dir = "output"
    image.save(temp_image_path)

    # Step 1: Image Segmentation
    segmented_image, segmented_objects = segment_image(temp_image_path)
    
    # Step 2: Object Extraction and Storage
    extracted_objects = extract_objects(segmented_image, segmented_objects)
    print(extracted_objects)

    
    # Step 4: Text/Data Extraction from Objects
    extracted_text = extract_text(extracted_objects, output_dir=os.path.join(output_dir, "extracted_objects"))
    
    # Step 5: Summarize Object Attributes
    summarized_attributes = summarize_attributes(extracted_objects, extracted_text)
    
    # Step 6: Data Mapping
    mapped_data = map_data(extracted_objects, extracted_text, summarized_attributes)

    # Clean up temporary files
    os.remove(temp_image_path)

    return mapped_data, extracted_objects

def draw_bounding_boxes(image, mapped_data):
    draw = ImageDraw.Draw(image)
    for obj in mapped_data:
        box = obj["box"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"ID: {obj['id']}", fill="red")
    return image

st.title("Image Segmentation and Analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            mapped_data, extracted_objects = process_image(image)

        st.success("Image processed successfully!")

        # Display annotated image
        annotated_image = draw_bounding_boxes(image.copy(), mapped_data)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        # Display results table
        st.subheader("Detected Objects")
        df = pd.DataFrame(mapped_data)
        st.dataframe(df)

        # Display all detected object images
        st.subheader("Detected Object Images")
        cols = st.columns(3)  # Adjust the number of columns as needed
        for i, obj in enumerate(extracted_objects):
            with cols[i % 3]:
                obj_image = Image.open(obj['path'])
                st.image(obj_image, caption=f"Object {obj['id']}", use_column_width=True)

        # Display individual object details
        st.subheader("Object Details")
        for obj in mapped_data:
            with st.expander(f"Object {obj['id']} - {obj['category']}"):
                st.image(Image.open(obj['path']), caption=f"Object {obj['id']}", use_column_width=True)
                st.write(f"Category: {obj['category']}")
                
                st.write(f"Summary: {obj['summary']}")

        # Clean up extracted object images
        for obj in extracted_objects:
            os.remove(obj['path'])

if __name__ == "__main__":
    if not os.path.exists("extracted_objects"):
        os.makedirs("extracted_objects")