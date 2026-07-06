from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "data/batches"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found. Check your DATA_PATH and that .md files exist.")
        return
    chunks = split_documents(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Preview a sample chunk
    if len(chunks) > 10:
        sample = chunks[10]
        print("\n--- Sample chunk ---")
        print(sample.page_content)
        print("Metadata:", sample.metadata)
        print("--------------------\n")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear existing DB to avoid stale data
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared existing ChromaDB.")

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_PATH
    )
    # Note: db.persist() is deprecated in newer chromadb versions — persists automatically
    print(f"Saved {len(chunks)} chunks to ChromaDB at '{CHROMA_PATH}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
