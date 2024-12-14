# LUMI FIN : Transforming Financial Insights with AI
_**An AI-Powered Financial Document Analysis Tool**_

## Overview
This is a powerful tool/approach designed to analyze and extract insights from company annual reports

## Prerequisites
- Python 3.8 or higher
- Required Python libraries (specified in `requirements.txt`)
- Annual report files in PDF format

## Clone Repository
```bash
git clone https://github.com/shoryasethia/Pulse-OAB.git
cd Pulse-OAB
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Use
### 1. Add Annual Reports
Place the `Annual Report (.pdf)` of a company in the `data/pdfs/` directory.

### 2. Run Scripts
#### a. Using Qwen Model
```bash
python app.py
```
#### b. Using Gemini Model
In `.env`, add your `GOOGLE_API_KEY`, if using `gemini.py`
```bash
python gemini.py
```


- **Output**: Results will be saved in the `output/` directory.

### 3. Knowledge Graph Interaction
#### a. Graph Response with Streamlit Interface
This launches a Streamlit application for a user-friendly Q&A chatbot experience.
```bash
streamlit run main.py
```
NOTE: To switch off debug mode, set `level = "info"` in `.streamlit/config.toml`
#### b. Graph Response in Terminal
Run the following command for graph-based terminal responses. Ensure you edit the query and select the appropriate mode of search (`local`, `naive`, `global`, `hybrid`) as per your requirements.
```bash
python graph-rag.py
```

## Project Structure
```
Pulse-OAB/
|-- .gitignore
|-- .streamlit
   |-- config.toml
   |-- secrets.toml
|-- app.py
|-- context_reduction.py
|-- csv-combine.ipynb
|-- csv2txt.py
|-- data
   |-- pdfs
      |-- 3M_2018_10K.pdf          # Replace this with your desired file
|-- gemini-mask.py
|-- gemini.py
|-- graph-rag.py
|-- images-response.py
|-- LICENSE
|-- lightrag
   |-- base.py
   |-- lightrag.py
   |-- llm.py
   |-- operate.py
   |-- prompt.py
   |-- storage.py
   |-- utils.py
   |-- __init__.py
   |-- __pycache__
|-- main.py
|-- model_loader.py
|-- output
   |-- extracted_tables
      |-- 3M_2018_10K
         |-- data.csv
         |-- table_page_8_box_1.png
         |-- table_page_13_box_2.png
         |-- ...
   |-- images
   |-- knowledge_graph
      |-- graph_chunk_entity_relation.graphml
      |-- kv_store_full_docs.json
      |-- kv_store_llm_response_cache.json
      |-- kv_store_text_chunks.json
      |-- lightrag.log
      |-- vdb_chunks.json
      |-- vdb_entities.json
      |-- vdb_relationships.json
   |-- parquet
      |-- extracted_texts_and_imgs.parquet
      |-- extracted_texts_and_imgs_with_responses.csv
   |-- txts
      |-- final_output.txt
|-- pdf_extractor.py
|-- README.md
|-- requirements.txt
|-- tables-response.py
|-- utils_logging.py
|-- __pycache__
```

## Project Rationale
Check out [Project Report](https://github.com/shoryasethia/Pulse-OAB/blob/main/Report.pdf) for details.

## To view results of `to-html`
> These code files are to create an .html page of past 10 years annual reports for faster analysis and data excess. This html page shows the annual report and have download buttons to download tables in excel format.

Page Url : [http://103.42.51.135/drereddy.html](http://103.42.51.135/drereddy.html)

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/shoryasethia/Pulse-OAB/blob/main/LICENSE) file for details.

## Support
For any issues or questions, please open an issue in the [GitHub repository](https://github.com/shoryasethia/Pulse-OAB/issues).


