import fitz
import glob
import os
import pandas as pd
import sys
import time
from typing import List, Any
from context_reduction import get_token_count
from utils_logging import utils_logger
from tqdm import tqdm  

from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from PIL import Image
import torch
import csv
from PIL import Image
from io import BytesIO

import subprocess

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info #type: ignore

#--------------------configurations---------------------------

tokens = 1024

IMAGES_DIR = r'output/images/'

INPUT_DATA = r'output/parquet/extracted_texts_and_imgs.parquet'

MANUALS_DIR = "data/pdfs"


pdf_dir = MANUALS_DIR

parquet_path = 'output/parquet/extracted_texts_and_imgs.parquet'

output_dir = 'output/parquet'

base_dir = 'output/extracted_tables'

#----------------------------------------------------------------

sys.setrecursionlimit(5000)


def get_pdf_chunks(pdf_reader: Any, max_tokens: int) -> List[List[Any]]:
    """
    Breaks a PDF document into chunks of pages.
    Each chunk contains pages that together have no more than max_tokens tokens.
    :param pdf_reader: The PDF document to be chunked.
    :param max_tokens: The maximum number of tokens that each chunk of pages can contain.
    :return: A list of chunks. Each chunk is a list of pages.
    """
    chunks = []
    pages = []
    n_tokens = 0
    for page in pdf_reader:
        page_tokens = get_token_count(page.get_text())
        if page_tokens + n_tokens > max_tokens:
            if len(pages):
                chunks.append(pages)
                pages = []
                n_tokens = 0
        n_tokens += page_tokens
        pages.append(page)
    if pages:
        chunks.append(pages)
    return chunks


def recoverpix(doc, item):
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask
    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])
        try:
            pix = fitz.Pixmap(pix0, mask)
        except:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"
        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
        }
    return doc.extract_image(xref)


def extract_images_from_pdf(pdf_reader, imgdir, chunk_number, store_to_folder=False, dimlimit=50, abssize=1024, relsize=0.0):
    """
    Extract images from a PDF document based on dimension, absolute size, and relative size limits.
    :param pdf_reader: The PDF document to extract images from.
    :param chunk_number: The number of the chunk from which to extract images.
    :param dimlimit: The minimum dimension an image must have to be extracted.
    :param abssize: The minimum absolute size an image must have to be extracted.
    :param relsize: The minimum relative size an image must have to be extracted.
    :return: A list of dictionaries, each containing the bytes of an image.
    """
    images = []
    xreflist = []
    img_list = pdf_reader.get_page_images(chunk_number - 1)
    for img in img_list:
        img_dict = {}
        xref = img[0]
        if xref in xreflist:
            continue
        width = img[2]
        height = img[3]
        if min(width, height) <= dimlimit:
            continue
        image = recoverpix(pdf_reader, img)
        colorspace = image["colorspace"]
        imgdata = image["image"]
        imgfile = os.path.join(imgdir, f"img%05i_p{chunk_number}.%s" % (xref, image["ext"]))
        if len(imgdata) <= abssize:
            continue
        if len(imgdata) / (width * height * colorspace) <= relsize:
            continue
        img_dict["image_bytes"] = imgdata
        img_dict["image_id"] = os.path.basename(imgfile)
        if store_to_folder:
            if not os.path.exists(imgdir):  # make subfolder if necessary
                os.mkdir(imgdir)
            fout = open(imgfile, "wb")
            fout.write(imgdata)
            fout.close()
        images.append(img_dict)
    return images


def create_dataframe_from_pdf(input_file: str, use_pages: bool, max_tokens: int, imgdir: str, pandas_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create a dataframe from a PDF document. Each row in the dataframe corresponds to an image in the document.
    :param input_file: The path of the PDF file.
    :param use_pages: If True, the PDF will be chunked into individual pages. Otherwise, it will be chunked into groups
    of pages with no more than max_tokens tokens.
    :param max_tokens: The maximum number of tokens each chunk can contain. Ignored if use_pages is True.
    :param pandas_df: Optional Dataframe from which to read the relevant pages that match the current document.
    :return: A dataframe where each row corresponds to an image from the PDF document.
    """
    docs = []
    imgdir = os.path.join(imgdir, os.path.basename(input_file).split('.')[0])
    with fitz.open(input_file) as pdf_reader:
        try:
            doc_title = ""
            if pdf_reader.metadata:
                doc_title = pdf_reader.metadata.get("title", "")
                if pdf_reader.metadata.get("subject", ""):
                    doc_title += " - " + pdf_reader.metadata.get("subject", "")
        except Exception as ex:
            utils_logger.warning(
                f"Could not parse pdf document (PDFReader error) {input_file}: {ex}"
            )
        try:
            # Get the pages from the pandas dataframe that match the current document
            if use_pages:
                if pandas_df is not None:
                    pages_to_extract = pandas_df[pandas_df['doc_id'] == os.path.basename(input_file)]['page_number'].tolist()
                    chunks = [[page] for i, page in enumerate(pdf_reader) if i + 1 in pages_to_extract]
                else:
                    chunks = [[page] for page in pdf_reader]
            else:
                chunks = get_pdf_chunks(pdf_reader, max_tokens)
        except Exception as ex:
            utils_logger.warning(f"Could not extract pages from pdf document (PDFReader error) {input_file}: {ex}")
        for chunk_number, chunk in enumerate(chunks, start=1):
            text_parts = [page.get_text() for page in chunk]
            text = " ".join(text_parts)
            first_page_number = chunk[0].number + 1
            last_page_number = chunk[-1].number + 1
            if first_page_number != last_page_number:
                page_number = f"{first_page_number}-{last_page_number}"
            else:
                page_number = str(first_page_number)
            images = extract_images_from_pdf(pdf_reader, imgdir, chunk[0].number+1, store_to_folder=True)
            if len(images) > 0:
                for image in images:
                    doc = {"doc_id": os.path.basename(input_file),
                           "doc_title": doc_title,
                           "page_number": page_number,
                           "text": text,
                           "url": f"{os.path.basename(input_file)}#page={first_page_number}",
                           "has_image": True,
                           "image_id": f"{os.path.basename(input_file).split('.')[0]}\\{image['image_id']}",
                           "image_bytes": image["image_bytes"]}
                    docs.append(doc)
            else:
                doc = {"doc_id": os.path.basename(input_file),
                       "doc_title": doc_title,
                       "page_number": page_number,
                       "text": text,
                       "url": f"{os.path.basename(input_file)}#page={first_page_number}",
                       "has_image": False,
                       "image_id": None,
                       "image_bytes": None}
                docs.append(doc)
    df = pd.DataFrame(docs)
    df.insert(loc=0, column='index', value=df.index)
    return df


def load_tabel_transformer():
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    return model, processor

# Function to convert PDF pages to images within a range
# Function to convert PDF pages to images within a range
def pdf_to_images(pdf_path, start_page=0, end_page=None):
    doc = fitz.open(pdf_path)
    images = []
    
    # Handle the case where end_page is not provided or exceeds the total number of pages
    end_page = min(end_page if end_page is not None else len(doc), len(doc))

    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # Adjust dpi for higher/lower resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    return images

# Function to detect tables in images
def detect_tables(images):
    table_results = []
    for image in images:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)

        # Store results
        table_results.append(results)
    return table_results

# Main function to extract tables from PDF or directory of PDFs
def extract_tables_from_pdf(pdf_path, start_page=0, end_page=None, threshold=0.8, 
                            output_dir="output/extracted_tables", padding_top_bottom=50, 
                            padding_left_right=20, delete_cache=True, verbose=None):
    
    # Check if pdf_path is a directory
    if os.path.isdir(pdf_path):
        # Process each PDF file in the directory
        for filename in os.listdir(pdf_path):
            if filename.endswith(".pdf"):
                pdf_file_path = os.path.join(pdf_path, filename)
                output_subdir = os.path.join(output_dir, os.path.splitext(filename)[0])
                csv_path = os.path.join(output_subdir, "data.csv")
                
                # Create a subdirectory for each PDF to store results
                os.makedirs(output_subdir, exist_ok=True)
                
                # Process the individual PDF
                _process_single_pdf(pdf_file_path, start_page, end_page, threshold, 
                                    output_subdir, csv_path, padding_top_bottom, 
                                    padding_left_right, delete_cache, verbose)
    else:
        # Single PDF case
        pdf_file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_subdir = os.path.join(output_dir, pdf_file_name)
        csv_path = os.path.join(output_subdir, "data.csv")
        
        # Create output directory for this single PDF
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process the single PDF
        _process_single_pdf(pdf_path, start_page, end_page, threshold, 
                            output_subdir, csv_path, padding_top_bottom, 
                            padding_left_right, delete_cache, verbose)

# Helper function to process a single PDF and save results
def _process_single_pdf(pdf_file_path, start_page, end_page, threshold, 
                        output_dir, csv_path, padding_top_bottom, 
                        padding_left_right, delete_cache, verbose):
    
    # Clear the output directory if delete_cache is True
    if delete_cache:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove directory if needed
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    # Prepare CSV file
    with open(csv_path, mode="w", newline="") as csv_file:
        fieldnames = ["index", "doc_id", "page_number", "has_table", "image_bytes"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Get images only for the specified page range
        images = pdf_to_images(pdf_file_path, start_page, end_page)
        table_results = detect_tables(images)
        
        tot_tables = 0  # Track the total number of detected tables
        doc_id = os.path.basename(pdf_file_path)  # Get document name
        
        # Print or process the detected tables
        for page_num, results in enumerate(table_results, start=start_page):
            has_table = False  # Initially set has_table as False
            
            for result in results:
                for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                    if score.item() >= threshold:
                        has_table = True
                        tot_tables += 1  # Increment the table counter
                        
                        # Convert box coordinates to integers
                        box = box.int().tolist()  # box is a tensor; convert to list
                        x0, y0, x1, y1 = box
                        # Add padding to the coordinates
                        x0 = max(0, x0 - padding_left_right)  # Ensure x0 doesn't go below 0
                        y0 = max(0, y0 - padding_top_bottom)   # Ensure y0 doesn't go below 0
                        x1 = min(images[page_num - start_page].width, x1 + padding_left_right)  # Ensure x1 doesn't go beyond image width
                        y1 = min(images[page_num - start_page].height, y1 + padding_top_bottom) # Ensure y1 doesn't go beyond image height
                        
                        # Crop the image using the bounding box coordinates with padding
                        cropped_image = images[page_num - start_page].crop((x0, y0, x1, y1))
                        
                        # Save the cropped image and get its bytes
                        table_image_path = os.path.join(output_dir, f"table_page_{page_num + 1}_box_{tot_tables}.png")
                        cropped_image.save(table_image_path)
                        
                        # Convert cropped image to bytes for CSV
                        img_byte_arr = BytesIO()
                        cropped_image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Write to CSV
                        writer.writerow({
                            "index": tot_tables,
                            "doc_id": doc_id,
                            "page_number": page_num + 1,
                            "has_table": has_table,
                            "image_bytes": img_bytes
                        })
                        
                        if verbose:
                            print(f"Detected {model.config.id2label[label.item()]} with score {score.item():.3f} at {box}. Saved as {table_image_path}")
            
            # If no table was detected, still log the page in CSV
            if not has_table:
                writer.writerow({
                    "index": tot_tables + 1,
                    "doc_id": doc_id,
                    "page_number": page_num + 1,
                    "has_table": has_table,
                    "image_bytes": None
                })
        
    # Final summary of extracted tables
    if verbose:
        print(f"Extraction complete for {doc_id}: {tot_tables} tables detected and saved in {output_dir}.")
    # else:
    #     print(f"Done. \nExtracted {tot_tables} tables in total for {doc_id}. All images saved at {output_dir}.")

def setup_model():
    """Initialize the model and processor"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-72B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
    return model, processor

#####################Tables###################################

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
                {"type": "text", "text": "Convert the following table into a descriptive paragraph. Include each row's numerical values and provide context for them, presenting the information in a coherent and flowing narrative form. Use complete sentences, and structure the paragraph to smoothly connect each row's details. Be sure to embed each numerical value within the text without skipping any information, creating a comprehensive summary that would be easy to read as long-form text. Here is the table"},
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
    generated_ids = model.generate(**inputs, max_new_tokens=tokens)
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


#######################Images################################
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
                    #Give me a very descriptive information present or explained in the image as paragraph of texts.
                    {"type": "text", "text": "Provide a thorough, detailed description of the image. Including all the minor details."},
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
        generated_ids = model.generate(**inputs, max_new_tokens=tokens)
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





if __name__ == "__main__":
    start_time = time.time()

    combined_df = pd.DataFrame()

    input_files = glob.glob(os.path.join(MANUALS_DIR, "*.pdf"))
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)

    for input_file in input_files:
        df = create_dataframe_from_pdf(
            input_file=input_file, 
            use_pages=True,
            max_tokens=0,
            imgdir=IMAGES_DIR,
            pandas_df=None
        )
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_parquet(INPUT_DATA, engine='pyarrow')
    
    model,processor = load_tabel_transformer()
        
    extract_tables_from_pdf(
        pdf_dir,
        threshold=0.88,
        output_dir="output/extracted_tables",
        padding_top_bottom=70,
        padding_left_right=40,
        delete_cache=True,
        verbose=False
    )

    model, processor = setup_model()
    
    print("qwen2 loaded successfully.")

    process_parquet_file(parquet_path, output_dir, model, processor)
    
    print("image-response.py completed successfully.")

    for pdf_folder in os.listdir(base_dir):
        pdf_folder_path = os.path.join(base_dir, pdf_folder)
        if os.path.isdir(pdf_folder_path):
            print(f"Processing folder: {pdf_folder}")
            process_pdf_folder(pdf_folder_path, model, processor)
    
    print("table-response.py completed successfully.")
    
    try:
        result_table = subprocess.run(['python', 'csv2txt.py'], check=True)
        print("csv2text.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running csv2txt.py: {e}")
        
    try:
        result_table = subprocess.run(['python', 'graph-rag.py'], check=True)
        print("graph-rag.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running graph-rag.py: {e}")

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))
