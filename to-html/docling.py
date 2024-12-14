from docling.document_converter import DocumentConverter

# List of URLs to process
urls = [
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240320.pdf",
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240319.pdf",
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240318.pdf",
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240317.pdf",
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240316.pdf",
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240315.pdf",
"https://www.bseindia.com/bseplus/AnnualReport/500124/5001240314.pdf"

]



converter = DocumentConverter()
# Iterate over each URL, process the document, and save results


count=2020

for url in urls:
    try:
        # Convert the document
        result = converter.convert(url)
        
        # Extract the filename from the URL to name the output file
        file_name = url.split("/")[-1]
        output_file = f"AnnualReport/drReddy/{count}.md"
        count=count-1

        # Save results to a text file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.document.export_to_markdown())
        
        print(f"Processed and saved: {output_file}")
    
    except Exception as e:
        print(f"Error processing {url}: {e}")

