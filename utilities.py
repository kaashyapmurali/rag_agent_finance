
import sqlite3
import os

def list_chroma_collections(PERSIST_DIR="./vector_store"):
    DB_PATH = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    print(f"Connected to {DB_PATH}")
    
    # Query to link collection names to their UUIDs (which are the folder names)
    try:
        cur.execute("SELECT id, name FROM collections")
        rows = cur.fetchall()
        
        print(f"{'COLLECTION NAME':<30} | {'FOLDER UUID (Directory)'}")
        print("-" * 70)
        
        for uuid, name in rows:
            print(f"{name:<30} | {uuid}")
            
    except sqlite3.OperationalError:
        print("Could not find 'collections' table. Ensure you are pointing to the correct chroma.sqlite3 file.")
    finally:
        con.close()