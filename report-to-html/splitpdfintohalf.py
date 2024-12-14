import fitz  # PyMuPDF
import os
from typing import Tuple, Optional

class PDFSplitError(Exception):
    """Custom exception for PDF splitting errors"""
    pass

def get_page_dimensions(page: fitz.Page) -> Tuple[float, float]:
    """
    Get the dimensions of a page
    
    Args:
        page: The PDF page object
    
    Returns:
        Tuple containing width and height
    """
    page_rect = page.rect
    return page_rect.width, page_rect.height

def is_page_empty(page: fitz.Page) -> bool:
    """
    Check if a page is empty
    
    Args:
        page: The PDF page object
    
    Returns:
        Boolean indicating if page is empty
    """
    text = page.get_text()
    images = page.get_images()
    return len(text.strip()) == 0 and len(images) == 0

def create_split_page(output_doc: fitz.Document, 
                     source_doc: fitz.Document,
                     page_num: int,
                     clip_rect: fitz.Rect,
                     width: float,
                     height: float) -> Optional[fitz.Page]:
    """
    Create a new page with content from the clipped region of the source page
    
    Args:
        output_doc: The output PDF document
        source_doc: The source PDF document
        page_num: Page number to process
        clip_rect: Rectangle defining the region to clip
        width: Width of the new page
        height: Height of the new page
    
    Returns:
        The created page object or None if creation fails
    """
    try:
        new_page = output_doc.new_page(width=width, height=height)
        new_page.show_pdf_page(new_page.rect, source_doc, page_num, clip=clip_rect)
        return new_page
    except ValueError as e:
        print(f"Warning: Failed to create split page for region {clip_rect} on page {page_num + 1}: {str(e)}")
        return None

def split_pdf(input_path: str, output_path: str, skip_empty: bool = True) -> None:
    """
    Split a PDF with two pages side by side into separate pages
    
    Args:
        input_path: Path to input PDF file
        output_path: Path to save output PDF file
        skip_empty: Whether to skip empty pages (default: True)
    
    Raises:
        PDFSplitError: If there's an error processing the PDF
    """
    if not os.path.exists(input_path):
        raise PDFSplitError(f"Input file not found: {input_path}")
    
    try:
        # Open the input PDF
        doc = fitz.open(input_path)
        print(f"Number of pages in input PDF: {len(doc)}")
        output_doc = fitz.open()
        
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                print(f"Processing page {page_num + 1}")
                
                # Skip empty pages if requested
                if skip_empty and is_page_empty(page):
                    print(f"Skipping empty page {page_num + 1}")
                    continue
                
                # Get page dimensions
                width, height = get_page_dimensions(page)
                half_width = width / 2
                
                # Create clip rectangles for left and right sides
                clip_rect_left = fitz.Rect(0, 0, half_width, height)
                clip_rect_right = fitz.Rect(half_width, 0, width, height)
                
                # Create split pages
                left_page = create_split_page(output_doc, doc, page_num, clip_rect_left, half_width, height)
                right_page = create_split_page(output_doc, doc, page_num, clip_rect_right, half_width, height)
                
                if not left_page and not right_page:
                    print(f"Warning: Both left and right splits failed for page {page_num + 1}")
                
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {str(e)}")
                continue
        
        # Save the output PDF
        output_doc.save(output_path)
        print(f"Output saved to {output_path}")
        
    except Exception as e:
        raise PDFSplitError(f"Error splitting PDF: {str(e)}")
    finally:
        if 'doc' in locals():
            doc.close()
        if 'output_doc' in locals():
            output_doc.close()

import os

def main():
    """Main function to process all PDFs in a directory"""
    input_directory = "/nfs/kundeshwar/surajOCR/annualReportFinalPipeline/drReddy/"  # Replace with the path to your directory containing PDFs
    output_directory = "/nfs/kundeshwar/surajOCR/annualReportFinalPipeline/drReddy/trimmed/"  # Replace with the desired output directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        # Iterate over all files in the directory
        for filename in os.listdir(input_directory):
            if filename.endswith(".pdf"):  # Process only PDF files
                input_pdf = os.path.join(input_directory, filename)
                output_pdf = os.path.join(output_directory, filename)
                
                print(f"Processing {input_pdf}...")
                split_pdf(input_pdf, output_pdf)
                print(f"Saved processed file to {output_pdf}")

    except PDFSplitError as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
