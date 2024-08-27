import os
import glob
import json
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

def process_pdfs_and_cache(input_folder, output_folder, strategy):
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
                    strategy=strategy,
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

if __name__ == "__main__":
    input_folder = "./data/farFull_seperatedParts"
    output_folder = "./cache"
    strategy = "auto"
    
    combined_content = process_pdfs_and_cache(input_folder, output_folder, strategy)
    print("Processing completed successfully.")