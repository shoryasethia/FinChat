import os
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

def setup_model():
    """Initialize the model and processor"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-72B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
    return model, processor

def generate_response_table(image_path, model, processor):
    """Generate response for a single image"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Please provide a detailed descriptive paragraph of all contents within the table image, "
                                            "including all formulas, headings, and values as text exactly as they appear in the table. "
                                            "Avoid summarizing; include every detail in the response."},
            ],
        }
    ]
    
    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def extract_page_number(image_name):
    """Extract page number from image filename"""
    # Assumes format: table_page_X_box_Y.png
    parts = image_name.split('_')
    try:
        return int(parts[2])
    except (IndexError, ValueError):
        return None

def process_pdf_folder(pdf_folder_path, model, processor):
    """Process all images in a PDF folder and update its CSV"""
    csv_path = os.path.join(pdf_folder_path, 'data.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize response column if it doesn't exist
    if 'response' not in df.columns:
        df['response'] = None
    
    # Process each image in the folder
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith('.png'):
            page_number = extract_page_number(filename)
            if page_number is not None:
                # Find matching row in CSV
                mask = (df['page_number'] == page_number) & (df['has_table'] == True)
                if any(mask):
                    image_path = os.path.join(pdf_folder_path, filename)
                    try:
                        response = generate_response_table(image_path, model, processor)
                        df.loc[mask, 'response'] = response
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    return df

def main():
    # Setup model and processor
    model, processor = setup_model()
    
    # Path to the extracted_tables directory
    base_dir = 'output/extracted_tables'
    
    # Process each PDF folder
    for pdf_folder in os.listdir(base_dir):
        pdf_folder_path = os.path.join(base_dir, pdf_folder)
        if os.path.isdir(pdf_folder_path):
            print(f"Processing folder: {pdf_folder}")
            process_pdf_folder(pdf_folder_path, model, processor)

if __name__ == "__main__":
    main()