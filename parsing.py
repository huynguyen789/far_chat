#PARSE DATA
import os
import glob
import json
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
from  langchain.schema import Document
from IPython.display import JSON, display, Markdown
from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# from  langchain.schema import Document
import json
from typing import Iterable
import logging


def process_pdfs_and_cache(input_folder, output_folder):
    # Load environment variables from a .env file
    load_dotenv()

    # Get unstructured API key
    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not unstructured_api_key:
        raise ValueError("UNSTRUCTURED_API_KEY environment variable not found")

    # Initialize the UnstructuredClient
    s = UnstructuredClient(api_key_auth=unstructured_api_key, server_url='https://redhorse-d652ahtg.api.unstructuredapp.io')

    # Create cache folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate cache file path
    folder_name = os.path.basename(os.path.normpath(input_folder))
    cache_file_path = os.path.join(output_folder, f'{folder_name}_combined_content.json')

    # Check if the combined content file already exists
    if os.path.exists(cache_file_path):
        print(f"Loading combined content from {cache_file_path}...")
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            combined_content = json.load(f)
    else:
        # Initialize a list to hold the combined content
        combined_content = []

        # Iterate through all PDF files in the directory
        for filename in glob.glob(os.path.join(input_folder, "*.pdf")):
            print(f"Processing {filename}...")
            with open(filename, "rb") as file:
                req = shared.PartitionParameters(
                    files=shared.Files(
                        content=file.read(),
                        file_name=filename,
                    ),
                    strategy="auto",
                )

                try:
                    res = s.general.partition(req)
                    # Append the parsed elements to the combined content list
                    combined_content.extend(res.elements)
                except SDKError as e:
                    print(f"Error processing {filename}: {e}")

        # Display length of combined content
        print(f"Combined content length: {len(combined_content)}")

        # Save combined content to the cache file
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_content, f)

        print(f"Combined content saved to {cache_file_path}")

    return combined_content

# Example usage:
combined_content = process_pdfs_and_cache("./data/test50pages", "./cache")

#PROCESS DATA


# Function to process and chunk the data
def process_data(combined_content):
    pdf_elements = dict_to_elements(combined_content)
    elements = chunk_by_title(pdf_elements, combine_text_under_n_chars=2000 , new_after_n_chars=4500, max_characters=5000, overlap=700)
    # elements = chunk_elements(pdf_elements, max_characters=5000, overlap=1000)
    documents = []
    for element in elements:
        metadata = element.metadata.to_dict()
        del metadata["languages"]
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))
    
    return documents

print(f"Chunking data...")
documents = process_data(combined_content)
print("Finished chunking data")
# print(documents[0])
# display(Markdown(documents[0].page_content))

# LOAD INTO VECTOR STORE


# Initialize the OpenAIEmbeddings class
embeddings = OpenAIEmbeddings()

# Initialize the Chroma vector store
#clear the vector store
print(f"Loading data into vector store")
vectorstore = FAISS.from_documents(documents, embeddings)
print("Finish loading data into vector store")

# Create a retriever
retriever = vectorstore.as_retriever(
    # search_type="similarity",
    search_kwargs={"k": 4})

query = "1.102-2 Performance standards"
print(f"Getting answer for: {query}")
answer = retriever.invoke(query)

# Calculate the total length of all page_content
total_length = sum(len(doc.page_content) for doc in answer)
print(f"Total length of all page_content combined: {total_length}")

answer

#display each retrieve document for debugging
for doc in answer:
    display(Markdown(doc.page_content))
    print("---------------------------------------------------")
    print("\n\n")