from datasets import load_dataset
import json
import os
from langchain_core.documents import Document


BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, "data")
INDEX_PATH = os.path.join(BASE_PATH, "vector_store")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

def ingest_and_save_data(num_samples=10, overwrite=True):
    destination_file = os.path.join(DATA_PATH, "earnings_transcripts.jsonl")
    if os.path.exists(destination_file):
        if not overwrite:
            print(f"Data already exists at {destination_file}. Skipping download.")
            return destination_file
        else:
            print(f"Overwriting existing data at {destination_file}.")
            os.remove(destination_file)

    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset("finosfoundation/EarningsCallTranscript", split="train", streaming=True)
    dataset = dataset.select_columns(['transcript', 'original_filename', 'segment_id'])

    if num_samples:
      print(f"Saving first {num_samples} records ...")
    else:
      print("Saving all records ...")

    with open(destination_file, "w") as f:
        for i, row in enumerate(dataset):
            if num_samples and i >= num_samples: break

            clean_record = {
                "id": i,
                "ticker": row['original_filename'].split('_')[0],
                "transcript": row['transcript'],
                "segment_id": row['segment_id']
            }
            f.write(json.dumps(clean_record) + "\n")

    print("Ingestion complete!")
    return destination_file

raw_data_path = ingest_and_save_data(num_samples=10, overwrite=True)

