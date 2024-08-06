def map_data(extracted_objects, extracted_text, summarized_attributes):
    mapped_data = []
    for obj, text, summary in zip(extracted_objects, extracted_text, summarized_attributes):
        mapped_data.append({
            "id": obj["id"],
            "path": obj["path"],
            "box": obj["box"],
            "category": text["text"],
            "summary": summary["summary"]
        })
    
    return mapped_data