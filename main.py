import pandas as pd
import json
import os
import time
from datasets import load_dataset
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Optional, Literal

# set up env variables
load_dotenv()
BASE_PATH = os.getcwd()

class TranscriptMetadataSearchQuery(BaseModel):  
    standalone_query: str = Field(
            ..., 
            description="The fully contextualized search query. If chat history exists, rephrase the user's last question to be standalone. E.g. 'How much did they make?' -> 'Apple revenue 2023'."
        )
    # We leave ticker as a string because the list might be too huge for a prompt
    ticker: Optional[str] = Field(
        None, 
        description="The stock ticker symbol (e.g., AAPL, MSFT). UPPERCASE."
    )
    year: Optional[int] = Field(
        None, 
        description="The 4-digit year (e.g., 2023)."
    )
    # strictly enforce the 1.0, 2.0 format using Literal
    quarter: Optional[Literal[1.0, 2.0, 3.0, 4.0]] = Field(
        None, 
        description="The fiscal quarter. MUST be one of: 1.0, 2.0, 3.0, 4.0"
    )
    sector: Optional[str] = Field(None, description="The market sector (e.g., Technology).")

class FinancialAgent:
    def __init__(self, collection_name = "earnings_call", data_dir_path=None, index_dir_path=None):
        self.DATA_DIR_PATH = os.path.join(BASE_PATH, "data") if not data_dir_path else data_dir_path
        self.INDEX_DIR_PATH = os.path.join(BASE_PATH, "vector_store") if not index_dir_path else index_dir_path
        self.ENTITIES_PATH = os.path.join(self.DATA_DIR_PATH, f"cached_entities_{collection_name}.json")
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
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        self._load_or_build_registry()

    def call_agent(self):
        chat_history = []
        
        # Pre-compile the structured output chain ONCE
        # This instructs the LLM to act as a "Query Refiner and Router"
        router_system_prompt = """
        You are a financial analyst helper.
        You are a search query optimizer for a financial analyst.
        1. Parse the user's latest question.
        2. If chat history exists, REFORMULATE the question to be standalone and explicit.
        3. EXTRACT metadata filters (Ticker, Year, Quarter) if mentioned.
        """
        
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", router_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        # This handles the consolidation of Step 1 and Step 2
        query_router = router_prompt | self.llm.with_structured_output(TranscriptMetadataSearchQuery)

        print("Agent Ready. (Type 'q' to quit)")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            # --- STEP 1: Single Call for Rephrasing + Extraction ---
            # We pass the history directly here.
            try:
                optimized_query = query_router.invoke({
                    "history": chat_history,
                    "question": user_input
                })
            except Exception as e:
                print(f"Router Error: {e}")
                continue
                
            print(f"User's Q: '{optimized_query.standalone_query}")
            print(f"Searching the DB based on these filters: {self.validate_and_clean_filters(optimized_query)} ...")

            # --- STEP 2: Retrieval ---
            valid_filters = self.validate_and_clean_filters(optimized_query)
            final_filter = self.build_chroma_filter(valid_filters)
            
            retriever = self.vector_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 5, 'filter': final_filter}
            )
            
            docs = retriever.invoke(optimized_query.standalone_query)
            print(f"Retrieved the most relevant {len(docs)} documents.")
            context_text = "\n\n".join([d.page_content for d in docs])

            # --- STEP 3: Streaming Generation ---
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful financial analyst. Answer based on the context provided. Keep it concise."),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            
            chain = final_prompt | self.llm
            
            print("AI: ", end="", flush=True)
            full_response = ""
            
            # Using .stream() creates the perception of instant speed
            for chunk in chain.stream({
                "context": context_text,
                "question": optimized_query.standalone_query # Use the cleaned query
            }):
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            
            print() # Newline after stream
            
            # Update history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=full_response))
            
            # Keep history short to prevent blowing up the Router's context window
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
    
    def download_and_save_data(self, num_samples=None, overwrite=False):
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

    def create_chunks_and_save_vector_db(self, ticker=None, batch_size=100, test=False, n_samples_test=10):
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
                if ticker and row['ticker'] != ticker:
                    continue

                metadata = {
                    "ticker": row.get('ticker') or 'UNKNOWN',
                    "year": int(row['year']) if row['year'] else 0,
                    "quarter": row.get('quarter') or 'UNKNOWN',
                    "sector": row.get('sector') or 'UNKNOWN',
                    "industry": row.get('industry') or 'UNKNOWN'
                }
                # 2. Check if it exists in the vector store, skip if yes
                unique_id = f"{metadata['ticker']}_{metadata['year']}_{metadata['quarter']}"
                if unique_id in self.processed_ids:
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"Skipped {skipped_count} existing reports so far...")
                    continue

                # 3. Add the doc to the vector store
                chunks = text_splitter.split_text(row['transcript'])
                
                for chunk in chunks:
                    batch_docs.append(Document(page_content=chunk, metadata=metadata))

                self.VALID_TICKERS.add(metadata['ticker'])
                self.VALID_SECTORS.add(metadata['sector'])
                self.processed_ids.add(unique_id)

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
        self._save_manifest()
        print(f"Processed {new_docs_count + skipped_count} chunks. Total Skipped: {skipped_count}, Total New: {new_docs_count}")

    def validate_and_clean_filters(self, query_data: TranscriptMetadataSearchQuery):
        clean_filters = {}

        # 1. Ticker Validation (Fuzzy matching or strict lookup)
        if query_data.ticker:
            # Simple strict check (convert to upper case)
            ticker_upper = query_data.ticker.upper()
            if ticker_upper in self.VALID_TICKERS:
                clean_filters['ticker'] = ticker_upper
            else:
                print(f"Warning: Didn't find '{query_data.ticker}' in the database. Ignoring.")

        # 2. Year Validation
        if query_data.year:
            clean_filters['year'] = query_data.year

        # 3. Quarter Validation (Pydantic usually catches this, but good to be safe)
        if query_data.quarter:
            clean_filters['quarter'] = query_data.quarter

        # 4. Sector/Industry Validation
        if query_data.sector and query_data.sector in self.VALID_SECTORS:
            clean_filters['sector'] = query_data.sector

        return clean_filters
    
    def build_chroma_filter(self, clean_filters: dict):
        if not clean_filters:
            return None

        conditions = []
        for key, value in clean_filters.items():
            conditions.append({key: value})
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {'$and': conditions}

    def _load_or_build_registry(self):
        """
        Hybrid Loader:
        1. Try to load from fast JSON file.
        2. If missing, scan the DB (slow) and build the file.
        """
        # 1. Fast Path: Try loading from JSON
        if os.path.exists(self.ENTITIES_PATH):
            try:
                with open(self.ENTITIES_PATH, 'r') as f:
                    data = json.load(f)
                    self.VALID_TICKERS = set(data.get('tickers', []))
                    self.VALID_SECTORS = set(data.get('sectors', []))
                    self.processed_ids = set(data.get('processed_ids', []))
                    print(f"Fast-loaded registry. {len(self.processed_ids)} reports tracked.")
                    return
            except Exception as e:
                print(f"JSON load failed ({e}). Falling back to DB Scan.")

        # 2. Slow Path: Scan DB (Logic from your old __get_existing_signatures)
        print("Scanning DB to rebuild registry (caching)...")
        
        # Fetch metadata only
        result = self.vector_store.get(include=["metadatas"])
        
        self.VALID_TICKERS = set()
        self.VALID_SECTORS = set()
        self.processed_ids = set()

        for meta in result['metadatas']:
            # Extract basic filters
            if meta.get('ticker'): self.VALID_TICKERS.add(meta['ticker'])
            if meta.get('sector'): self.VALID_SECTORS.add(meta['sector'])
            
            # Create Unique ID: TICKER_YEAR_QUARTER
            # We use strings because they save to JSON easier than tuples
            sig = f"{meta.get('ticker')}_{meta.get('year')}_{meta.get('quarter')}"
            self.processed_ids.add(sig)

        # 3. Save it so we don't have to scan next time
        self.__save_manifest()
        print(f"Registry rebuilt from DB. Found {len(self.processed_ids)} unique reports.")

    def _save_manifest(self):
        """Helper to save the registry to JSON."""
        with open(self.ENTITIES_PATH, 'w') as f:
            json.dump({
                "tickers": list(self.VALID_TICKERS),
                "sectors": list(self.VALID_SECTORS),
                "processed_ids": list(self.processed_ids)
            }, f)


if __name__ == "__main__":
    # set parameters
    test = False
    overwrite = False

    agent = FinancialAgent(collection_name='earnings_call')
    agent.download_and_save_data(num_samples=None, overwrite=overwrite)
    agent.create_chunks_and_save_vector_db(ticker=None, test=test, batch_size=70, n_samples_test=2000)
    agent.call_agent()