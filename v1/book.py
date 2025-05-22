from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
import json
from typing import List
from pypdf import PdfReader
import logging
import argparse
from datetime import datetime


CONFIG = {
    "QDRANT_URL": "",
    "API_KEY": "",  
    "OUTPUT_FOLDER": r"D:\Downloads\All\op",
    "COLLECTION_NAME": "pdf_documents1",
    "DEFAULT_INPUT_PATH": r"D:\Downloads\All\bladder.pdf" 
}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_components():
    """Initialize Qdrant client and embedding model"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = QdrantClient(
        url=CONFIG["QDRANT_URL"],
        api_key=CONFIG["API_KEY"]
    )
    return model, client

def setup_collection(client):
    """Create collection if it doesn't exist"""
    try:
        client.create_collection(
            collection_name=CONFIG["COLLECTION_NAME"],
            vectors_config=models.VectorParams(
                size=384,  
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0
            )
        )
        logger.info("Created new collection")
    except Exception as e:
        logger.info(f"Using existing collection: {str(e)}")

def get_pdf_files(input_path: str) -> List[str]:
    """Returns list of PDF files from a file or folder."""
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        return [input_path]
    elif os.path.isdir(input_path):
        return [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith('.pdf')
        ]
    else:
        raise ValueError("Input must be a PDF file or a folder containing PDFs")

def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extract text from PDF with page tracking"""
    try:
        reader = PdfReader(pdf_path)
        pages_content = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages_content.append({
                    "page_number": page_num + 1,  # 1-indexed pages
                    "content": text
                })
                
        return {
            "total_pages": len(reader.pages),
            "pages": pages_content
        }
    except Exception as e:
        logger.error(f"Failed to read {pdf_path}: {str(e)}")
        return {"total_pages": 0, "pages": []}

def save_embeddings_locally(data: list):
    """Save embeddings to JSON file with timestamp"""
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"embeddings_backup_{timestamp}.json"
    output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved embeddings backup to {output_path}")

def process_pdfs(input_path: str):
    """Main processing function with page tracking"""
    model, client = initialize_components()
    setup_collection(client)
    
    try:
        pdf_files = get_pdf_files(input_path)
    except ValueError as e:
        logger.error(str(e))
        return

    if not pdf_files:
        logger.error("No PDF files found at the specified path")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    points = []
    all_embeddings = []
    point_id = 1
    
    for pdf_path in pdf_files:
        try:
            filename = os.path.basename(pdf_path)
            book_title = os.path.splitext(filename)[0]  # Extract book title from filename
            logger.info(f"Processing book: {book_title}")
            
            extracted_data = extract_text_from_pdf(pdf_path)
            if not extracted_data["pages"]:
                logger.warning(f"No text extracted from {filename}")
                continue
                
            total_pages = extracted_data["total_pages"]
            
            for page_data in extracted_data["pages"]:
                page_number = page_data["page_number"]
                page_content = page_data["content"]
                
                # Split page text into chunks (smaller chunks for better retrieval)
                chunks = [page_content[i:i+800] for i in range(0, len(page_content), 800)]
                
                for chunk_num, chunk in enumerate(chunks):
                    embedding = model.encode(chunk)
                    
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk,
                            "book_title": book_title,
                            "filename": filename,
                            "page_number": page_number,
                            "total_pages": total_pages,
                            "chunk_num": chunk_num,
                            "total_chunks_in_page": len(chunks),
                            "file_path": pdf_path
                        }
                    ))
                    
                    all_embeddings.append({
                        "id": point_id,
                        "text": chunk,
                        "embedding": embedding.tolist(),
                        "metadata": {
                            "book_title": book_title,
                            "filename": filename,
                            "page_number": page_number,
                            "chunk_num": chunk_num,
                            "file_path": pdf_path
                        }
                    })
                    
                    point_id += 1
                
        except Exception as e:
            logger.error(f"Failed processing {pdf_path}: {str(e)}")
    
    if points:
        # Upload to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=CONFIG["COLLECTION_NAME"],
                points=points[i:i+batch_size],
                wait=True
            )
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        
        # Save local backup
        save_embeddings_locally(all_embeddings)
        
        logger.info(f"""
        Processing complete!
        PDFs processed: {len(pdf_files)}
        Total chunks: {len(points)}
        Collection: {CONFIG["COLLECTION_NAME"]}
        Qdrant URL: {CONFIG["QDRANT_URL"]}
        Local backup saved to: {CONFIG["OUTPUT_FOLDER"]}
        """)
    else:
        logger.error("No valid PDF content found to process")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs into Qdrant vector database.")
    parser.add_argument(
        "--path", 
        type=str, 
        help="Path to a PDF file or folder (overrides default)"
    )
    args = parser.parse_args()

    
    if args.path:
        target_path = args.path
    elif os.path.exists(CONFIG["DEFAULT_INPUT_PATH"]):
        target_path = CONFIG["DEFAULT_INPUT_PATH"]
        logger.info(f"Using default path: {target_path}")
    else:
        
        while True:
            user_path = input(
                "Enter path to a PDF file or folder (or 'exit' to quit): "
            ).strip()
            if user_path.lower() in ('exit', 'quit'):
                exit()
            if os.path.exists(user_path):
                target_path = user_path
                break
            logger.error("Path does not exist. Please try again.")

    process_pdfs(target_path)