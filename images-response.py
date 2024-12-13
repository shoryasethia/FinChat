import os
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm  # For progress tracking

def setup_model():
    """Initialize the model and processor"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-72B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
    return model, processor

def generate_response_image(image_path, model, processor):
    """Generate response for a single image"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Give me a very descriptive information present or explained in the image as paragraph of texts."},
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
    except Exception as e:
        print(f"Error generating response for {image_path}: {str(e)}")
        return None

def fix_image_path(image_id):
    """Fix image path by replacing backslashes with forward slashes"""
    # Replace backslashes with forward slashes
    fixed_id = image_id.replace('\\', '/')
    # Remove any leading/trailing slashes
    fixed_id = fixed_id.strip('/')
    return fixed_id

def process_parquet_file(parquet_path, output_dir, model, processor):
    """Process parquet file and generate responses for images"""
    # Read parquet file
    print("Reading parquet file...")
    df = pd.read_parquet(parquet_path)
    
    # Convert has_image to boolean if it's not already
    df['has_image'] = df['has_image'].astype(bool)
    
    # Initialize response column
    df['image_response'] = None
    
    # Process only rows where has_image is True
    image_rows = df[df['has_image'] == True]
    print(f"Found {len(image_rows)} images to process")
    
    # Process each image
    for idx in tqdm(image_rows.index, desc="Processing images"):
        image_id = df.loc[idx, 'image_id']
        if pd.notna(image_id):
            # Fix image path
            fixed_image_id = fix_image_path(image_id)
            image_path = f'output/images/{fixed_image_id}'
            
            # Check if image file exists
            if os.path.exists(image_path):
                response = generate_response_image(image_path, model, processor)
                df.loc[idx, 'image_response'] = response
            else:
                # Try alternative path formats
                alt_paths = [
                    f'output/images/{fixed_image_id}',
                    f'output/images/1-s2/{os.path.basename(fixed_image_id)}',
                    f'output/images/1-s2/{os.path.basename(image_id)}'
                ]
                
                found = False
                for path in alt_paths:
                    if os.path.exists(path):
                        response = generate_response_image(path, model, processor)
                        df.loc[idx, 'image_response'] = response
                        found = True
                        break
                
                if not found:
                    print(f"Image not found in any of these locations:")
                    for path in alt_paths:
                        print(f"- {path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    output_path = os.path.join(output_dir, 'extracted_texts_and_imgs_with_responses.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Print some statistics
    total_images = len(image_rows)
    processed_images = df['image_response'].notna().sum()
    print(f"\nProcessing Summary:")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed images: {processed_images}")
    print(f"Failed to process: {total_images - processed_images}")
    
    # Print sample of directory structure
    print("\nChecking directory structure...")
    try:
        base_dir = 'output/images'
        if os.path.exists(base_dir):
            print(f"\nContents of {base_dir}:")
            for root, dirs, files in os.walk(base_dir, topdown=True, maxdepth=2):
                level = root.replace(base_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                if level < 1:  # Only show files for first two levels
                    subindent = ' ' * 4 * (level + 1)
                    for f in files[:5]:  # Show only first 5 files
                        print(f"{subindent}{f}")
                    if len(files) > 5:
                        print(f"{subindent}...")
    except Exception as e:
        print(f"Error while checking directory structure: {str(e)}")

def main():
    # Setup paths
    parquet_path = 'output/parquet/extracted_texts_and_imgs.parquet'
    output_dir = 'output/parquet'
    
    # Setup model and processor
    print("Setting up model and processor...")
    model, processor = setup_model()
    
    # Process parquet file
    df = process_parquet_file(parquet_path, output_dir, model, processor)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()