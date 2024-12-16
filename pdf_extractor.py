import fitz
import glob
import os
import pandas as pd
import sys
import time
from typing import List, Any
from context_reduction import get_token_count
from utils_logging import utils_logger

#configurations

IMAGES_DIR = r'output/images/'

INPUT_DATA = r'output/parquet/extracted_texts_and_imgs.parquet'


MANUALS_DIR = "data/pdfs"



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
    

    
    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))