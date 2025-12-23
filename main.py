import pandas as pd
import json
import os
import time
from datasets import load_dataset
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# set up env variables
load_dotenv()
BASE_PATH = os.getcwd()


class FinancialAgent:
    def __init__(self, collection_name = "earnings_call", data_dir_path=None, index_dir_path=None):
        self.DATA_DIR_PATH = os.path.join(BASE_PATH, "data") if not data_dir_path else data_dir_path
        self.INDEX_DIR_PATH = os.path.join(BASE_PATH, "vector_store") if not index_dir_path else index_dir_path
        os.makedirs(self.DATA_DIR_PATH, exist_ok=True)
        os.makedirs(self.INDEX_DIR_PATH, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document"
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.INDEX_DIR_PATH # persist data locally
        )

    def download_and_save_data(self, num_samples=10, overwrite=False):
        self.RAW_DATA_FILE_PATH = os.path.join(self.DATA_DIR_PATH, "earnings_transcripts.jsonl")
        if os.path.exists(self.RAW_DATA_FILE_PATH):
            if not overwrite:
                print(f"Data already exists at {self.RAW_DATA_FILE_PATH}. Skipping download.")
                return
            print(f"Overwriting existing data at {self.RAW_DATA_FILE_PATH}.")
            os.remove(self.RAW_DATA_FILE_PATH)

        print("Downloading dataset from Hugging Face...")
        dataset = load_dataset("glopardo/sp500-earnings-transcripts", split="train", streaming=True)

        if num_samples:
            print(f"Saving first {num_samples} records ...")
        else:
            print("Saving all records ...")

        with open(self.RAW_DATA_FILE_PATH, "w") as f:
            count = 0
            for row in dataset:
                if num_samples and count >= num_samples: break
                
                clean_record = {
                    "id": count,
                    "ticker": row.get('ticker', 'UNKNOWN'),
                    "sector": row.get('sector', 'UNKNOWN'),
                    "industry": row.get('industry', 'UNKNOWN'),
                    "datacqtr": row.get('datacqtr', 'UNKNOWN'),
                    "year": row.get('year', 'UNKNOWN'),
                    "quarter": row.get('quarter', 'UNKNOWN'),
                    "transcript": row.get('transcript', '')
                }

                # Filter out empty transcripts to keep quality high
                if len(clean_record['transcript']) > 10:
                    f.write(json.dumps(clean_record) + "\n")
                    count += 1
                    
                if count % 2000 == 0:
                    print(f"Processed {count} records...")
        print("Download complete!")

    def create_chunks_and_save_vector_db(self, batch_size=100, test=False, n_samples_test=10):
        batch_docs = []
        new_docs_count = 0
        skipped_count = 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        start_time = time.time()
        with open(self.RAW_DATA_FILE_PATH, "r") as f:
            for line in f:
                # 1. Read entry and transcripts
                row = json.loads(line)
                metadata = {
                    "ticker": row.get('ticker') or 'UNKNOWN',
                    "year": int(row['year']) if row['year'] else 0,
                    "quarter": row.get('quarter') or 'UNKNOWN',
                    "sector": row.get('sector') or 'UNKNOWN',
                    "industry": row.get('industry') or 'UNKNOWN'
                }
                # 2. Check if it exists in the vector store, skip if yes
                existing_data = self.vector_store.get(
                                    where={
                                        "$and": [
                                            {"ticker": {"$eq": metadata['ticker']}},
                                            {"year": {"$eq": metadata['year']}},
                                            {"quarter": {"$eq": metadata['quarter']}}
                                        ]
                                    },
                                    limit=1 
                )
                if len(existing_data['ids']) > 0:
                    # print(f"Skipping {metadata['ticker']} {metadata['year']} {metadata['quarter']} - Already indexed.")
                    skipped_count += 1
                    continue
                # 3. Add the doc to the vector store
                chunks = text_splitter.split_text(row['transcript'])
                
                for chunk in chunks:
                    batch_docs.append(Document(page_content=chunk, metadata=metadata))
                if len(batch_docs) >= batch_size or (test and new_docs_count == n_samples_test):
                    self.vector_store.add_documents(batch_docs)
                    new_docs_count += len(batch_docs)
                    print(f"Flushed {len(batch_docs)} chunks. Total Added: {new_docs_count}. Skipped so far: {skipped_count}. (Time elapsed: {time.time() - start_time:.2f}s)")
                    batch_docs = []

                # 4. Test function to break out of the loop if more than 
                if test and new_docs_count >= n_samples_test:
                    break
        if batch_docs:
            self.vector_store.add_documents(batch_docs)
            new_docs_count += len(batch_docs)
            print(f"Flushing remaing {len(batch_docs)} chunks.")
        print(f"Processed {new_docs_count + skipped_count} chunks. Total Skipped: {skipped_count}, Total New: {new_docs_count}")

if __name__ == "__main__":
    # set parameters
    test = False

    agent = FinancialAgent()
    agent.download_and_save_data(num_samples=None, overwrite=test)
    agent.create_chunks_and_save_vector_db(test=test, n_samples_test=10)
    print("Done!")