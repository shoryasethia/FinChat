import pandas as pd
import os

# Initialize an empty list to hold DataFrames
dataframes = []

# Define the directories
tables_dir = 'output/extracted_tables'

# Iterate through each folder in the tables directory
for folder_name in os.listdir(tables_dir):
    folder_path = os.path.join(tables_dir, folder_name)
    
    # Check if the path is a directory
    if os.path.isdir(folder_path):
        csv_path = os.path.join(folder_path, 'data.csv')
        
        # Check if the data.csv file exists
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df = df.drop(columns=['index'])
            dataframes.append(df)

# Concatenate all the DataFrames into a single DataFrame
df1 = pd.concat(dataframes, ignore_index=True)

# Load the second DataFrame
df2 = pd.read_csv('output/parquet/extracted_texts_and_imgs_with_responses.csv')
df2 = df2.drop(columns=['index', 'image_id'])

# Concatenate df1 and df2 by page_number
concatenated_df1 = df1.groupby('page_number')['response'].agg(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
concatenated_df2 = df2.groupby('page_number').agg({
    'text': lambda x: ' '.join(x.dropna().astype(str)),  # Concatenate text
    'image_response': lambda x: ' '.join(x.dropna().astype(str))  # Concatenate image_response
}).reset_index()

# Merge the two DataFrames on 'page_number'
final_df = pd.merge(concatenated_df2, concatenated_df1, on='page_number', how='outer')

# Select the desired columns for the final DataFrame
final_df = final_df[['page_number', 'text', 'response', 'image_response']]

# Define output directory and file
output_dir = 'output/txts/'
output_file = os.path.join(output_dir, 'final_output.txt')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Write to the .txt file
with open(output_file, 'w') as f:
    for index, row in final_df.iterrows():
        f.write(f"{row['text']} {row['response']} {row['image_response']}\n\n")  # Two newlines for space between entries
