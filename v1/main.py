from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from typing import List, Optional
from fastapi.responses import JSONResponse
import shutil
import uvicorn
import openai
import uuid
import json
import os
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import shutil
import logging
from openai import APIError, APIConnectionError, APITimeoutError
from starlette.requests import Request
from starlette.datastructures import UploadFile as StarletteUploadFile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "QDRANT_URL": os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
    "API_KEY": "",
    "UPLOAD_FOLDER": "./uploaded_books",
    "DEFAULT_COLLECTION": "all_books",
    "HISTORY_FILE": "chat_history.json",
    "OLLAMA_API_URL": os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/")
}

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI System",
    # Set maximum upload size to approximately 1GB (adjust as needed)
    max_request_size=1024*1024*1024
)

# Add a custom middleware to handle large file uploads
@app.middleware("http")
async def add_custom_header(request: Request, call_next):
    # Increase the default body limit (same as max_request_size above)
    request._body_size_limit = 1024*1024*1024  # 1GB
    response = await call_next(request)
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect slashes middleware
@app.middleware("http")
async def redirect_slashes(request: Request, call_next):
    if request.url.path.endswith("/"):
        return RedirectResponse(url=request.url.path.rstrip("/"))
    response = await call_next(request)
    return response

# Create directories
os.makedirs(CONFIG["UPLOAD_FOLDER"], exist_ok=True)

# Initialize clients and models
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

qdrant_client = QdrantClient(
    url=CONFIG["QDRANT_URL"],
    api_key=CONFIG["API_KEY"]
)

embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

VECTOR_SIZE = 384

# Setup collection
def setup_collection(collection_name):
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0
            )
        )
        logger.info(f"Created new collection: {collection_name}")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"Using existing collection: {collection_name}")
            return True
        logger.error(f"Failed to create collection: {str(e)}")
        return False

# Ensure default collection exists
setup_collection(CONFIG["DEFAULT_COLLECTION"])

# System prompts
MEDICAL_PROMPT = """
You are Medibot, a medical AI assistant for accurate, evidence-based medical information.
- Respond only to medical or health-related questions.
- Format responses in Markdown with:
  - A main heading (# [Topic])
  - Subheadings (## [Section])
  - Bullet points (- [Detail])
  - A conclusion (## Conclusion)

IMPORTANT CITATION RULES:
- You must include citations for every fact or statement from the provided sources
- Citations should be in the format: [Book Title, Page X] at the end of sentences
- When a point comes from multiple sources, include all relevant citations
- Do not make claims without supporting evidence from the sources
- If sources contradict, note the discrepancies and cite all relevant sources

For non-medical questions, respond with: "I'm Medibot, specialized in medical information. Please ask a health-related question."

Avoid direct medical advice beyond general knowledge.
"""

GENERAL_PROMPT = """
You are Medibot, a friendly and knowledgeable general-purpose chatbot. 
- Answer any type of question clearly and concisely.
- Provide helpful and accurate information in a conversational style.
- Do not use database context unless explicitly requested.
- No specific formatting is required, but keep responses engaging and natural.
"""

# Chat state
state = {
    "messages": [{"role": "system", "content": MEDICAL_PROMPT}],
    "current_session_id": str(uuid.uuid4()),
    "medical_mode": True
}

# Request/response models
class ChatRequest(BaseModel):
    message: str

class ToggleRequest(BaseModel):
    medical_mode: bool

class BookInfo(BaseModel):
    book_id: str
    title: str
    filename: str
    pages: int
    chunks: int
    upload_date: str

class BookUploadResponse(BaseModel):
    status: str
    book_id: str
    title: str
    pages: int
    chunks: int

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.3

class BookUpdateRequest(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None

# Helper functions
def ensure_history_file():
    if not os.path.exists(CONFIG["HISTORY_FILE"]):
        with open(CONFIG["HISTORY_FILE"], "w") as f:
            json.dump([], f)

def get_embedding(text):
    try:
        embedding = list(embedding_model.embed([text]))[0]
        return embedding.tolist()
    except Exception as e:
        return [0.0] * VECTOR_SIZE

def load_chat_history():
    try:
        ensure_history_file()
        with open(CONFIG["HISTORY_FILE"], "r") as f:
            history = json.load(f)
        history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:5]
        return history
    except Exception as e:
        return []

def save_chat_session(session_id, timestamp, messages):
    try:
        ensure_history_file()
        with open(CONFIG["HISTORY_FILE"], "r") as f:
            history = json.load(f)
        session = {
            "session_id": session_id,
            "timestamp": timestamp,
            "messages": messages
        }
        history = [s for s in history if s["session_id"] != session_id]
        history.append(session)
        with open(CONFIG["HISTORY_FILE"], "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        pass

def search_qdrant(query):
    try:
        collection_name = CONFIG["DEFAULT_COLLECTION"]
        
        # Log the search query for debugging
        logger.info(f"Searching for: {query}")
        
        # Try multiple search approaches
        query_vector = get_embedding(query)
        
        # First search with higher threshold for precision
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=8,
            with_payload=True,
            score_threshold=0.5  # Higher threshold for better quality
        )

        # If no results, try again with lower threshold
        if not search_result:
            logger.info("No results with high threshold, trying lower threshold")
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=8,
                with_payload=True,
                score_threshold=0.1  # Lower threshold to get more results
            )

        if not search_result:
            logger.warning("No relevant information found with any threshold")
            return "No relevant information found in the medical books."

        # Log what we found for debugging
        logger.info(f"Found {len(search_result)} relevant chunks")
        
        # Format results with source attribution
        context_entries = []
        for point in search_result:
            text = point.payload.get("text", "")
            book_title = point.payload.get("book_title", "Unknown Source")
            page_number = point.payload.get("page_number", "Unknown")
            
            logger.debug(f"Result from: {book_title}, Page {page_number}, Score: {point.score:.2f}")
            
            if text:
                formatted_entry = (
                    f"[SOURCE: {book_title}, Page {page_number}]\n"
                    f"{text}\n"
                    f"[Relevance: {point.score:.2f}]"
                )
                context_entries.append(formatted_entry)
                
        return "\n\n".join(context_entries)
    except Exception as e:
        logger.error(f"Error searching medical information: {str(e)}")
        return f"Error searching medical information: {str(e)}"

def retry_operation(operation, max_retries=3):
    """Retry an operation with exponential backoff"""
    import time
    retries = 0
    while retries < max_retries:
        try:
            return operation()
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise
            wait_time = 2 ** retries  # Exponential backoff
            logger.warning(f"Operation failed, retrying in {wait_time} seconds... ({retries}/{max_retries})")
            time.sleep(wait_time)

def validate_pdf(file_path):
    """Validate that the file is actually a PDF"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            # Check for PDF header signature
            if header != b'%PDF-':
                return False
        return True
    except:
        return False

class UploadStatus:
    def __init__(self):
        self.uploads = {}
        
    def start_upload(self, upload_id):
        self.uploads[upload_id] = {
            "status": "in_progress",
            "progress": 0,
            "total_size": 0,
            "current_size": 0
        }
        
    def update_progress(self, upload_id, current, total):
        if upload_id in self.uploads:
            self.uploads[upload_id]["current_size"] = current
            self.uploads[upload_id]["total_size"] = total
            self.uploads[upload_id]["progress"] = int(100 * current / total) if total > 0 else 0
            
    def complete_upload(self, upload_id):
        if upload_id in self.uploads:
            self.uploads[upload_id]["status"] = "completed"
            self.uploads[upload_id]["progress"] = 100
            
    def get_status(self, upload_id):
        return self.uploads.get(upload_id, {"status": "not_found"})

# Initialize the upload status tracker
upload_tracker = UploadStatus()

# Add an endpoint to check upload status
@app.get("/ai/upload-status/{upload_id}")
async def get_upload_status(upload_id: str):
    return upload_tracker.get_status(upload_id)

from pypdf import PdfReader
def is_encrypted_pdf(file_path):
    """Check if a PDF is encrypted"""
    try:
        reader = PdfReader(file_path)
        return reader.is_encrypted
    except:
        # If we can't even open it, consider it problematic
        return True

def extract_text_from_pdf(pdf_path: str) -> dict:
    """Enhanced PDF text extraction with better error handling and debugging"""
    try:
        from pypdf import PdfReader
        logger.info(f"Opening PDF file: {pdf_path}")
        
        # Check file exists and is readable
        if not os.path.exists(pdf_path):
            logger.error(f"File does not exist: {pdf_path}")
            return {"total_pages": 0, "pages": []}
            
        # Check if file is actually a PDF
        with open(pdf_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                logger.error(f"File is not a valid PDF: {pdf_path}")
                return {"total_pages": 0, "pages": []}
        
        reader = PdfReader(pdf_path)
        pages_content = []
        
        logger.info(f"PDF has {len(reader.pages)} pages")
        
        # Try multiple extraction methods if needed
        for page_num, page in enumerate(reader.pages):
            try:
                # First try the standard extraction
                text = page.extract_text()
                
                # If that fails, try alternative methods
                if not text or not text.strip():
                    # Try to extract structured content if available
                    logger.debug(f"Standard extraction failed for page {page_num+1}, trying alternative methods")
                    
                    # Check if we can get text from annotations or form fields
                    if hasattr(page, 'annotations'):
                        for annot in page.annotations:
                            if hasattr(annot, 'contents') and annot.contents:
                                text += annot.contents + "\n"
                
                if text and text.strip():  # Only add non-empty pages
                    pages_content.append({
                        "page_number": page_num + 1,  # 1-indexed pages
                        "content": text
                    })
                    logger.debug(f"Extracted {len(text)} characters from page {page_num+1}")
                else:
                    logger.warning(f"No text extracted from page {page_num+1}")
            except Exception as page_error:
                logger.warning(f"Error extracting text from page {page_num+1}: {str(page_error)}")
                # Continue to next page even if this one fails
        
        # If no text was extracted, try a more aggressive approach
        if not pages_content and reader.pages:
            logger.warning(f"Standard text extraction failed for {pdf_path}, trying more aggressive methods")
            try:
                import subprocess
                import tempfile
                
                # Try using pdftotext if available (external tool)
                logger.info("Attempting extraction with external tools")
                
                # Create temp file for output
                with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                    temp_path = temp.name
                
                # Try using pdftotext from poppler if available
                try:
                    result = subprocess.run(['pdftotext', pdf_path, temp_path], 
                                           check=True, capture_output=True)
                    
                    # Read the extracted text
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    # Split by pages (crude approximation)
                    text_pages = text.split('\f')
                    
                    for i, page_text in enumerate(text_pages):
                        if page_text.strip():
                            pages_content.append({
                                "page_number": i + 1,
                                "content": page_text
                            })
                    
                    logger.info(f"External extraction found {len(pages_content)} pages with content")
                    
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("External PDF tools not available")
                
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
            except Exception as fallback_error:
                logger.warning(f"Aggressive extraction also failed: {str(fallback_error)}")
        
        return {
            "total_pages": len(reader.pages),
            "pages": pages_content
        }
    except Exception as e:
        logger.error(f"Failed to read {pdf_path}: {str(e)}")
        return {"total_pages": 0, "pages": []}

# Add this to your process_book function to debug PDF extraction
def process_book(pdf_path: str, book_id: str, title: str = None) -> dict:
    """Process a single book and add to vector database with robust error handling"""
    try:
        # First, test if we can extract any text from the PDF
        test_extraction = ""
        try:
            # Simple test to see if we can get any text
            with open(pdf_path, 'rb') as f:
                from pypdf import PdfReader
                reader = PdfReader(f)
                for page in reader.pages[:3]:  # Try first 3 pages
                    text = page.extract_text()
                    if text and len(text) > 10:
                        test_extraction = text[:200]  # Save sample for logging
                        break
        except Exception as ex:
            logger.error(f"Basic PDF test extraction failed: {str(ex)}")
            
        if not test_extraction:
            logger.error(f"❌ CRITICAL: Cannot extract any text from {pdf_path}")
            logger.error("This is likely why your collection is empty")
            
            # Try to analyze the file structure
            try:
                file_size = os.path.getsize(pdf_path)
                with open(pdf_path, 'rb') as f:
                    header = f.read(1024)
                logger.error(f"File size: {file_size} bytes, Header starts with: {header[:20]}")
            except Exception as ex:
                logger.error(f"File analysis failed: {str(ex)}")
                
            # Return minimal info without further processing
            return {
                "book_id": book_id,
                "title": title or os.path.splitext(os.path.basename(pdf_path))[0],
                "filename": os.path.basename(pdf_path),
                "pages": 0, 
                "chunks": 0,
                "error": "Text extraction failed completely"
            }
        else:
            logger.info(f"✅ PDF basic text extraction successful: \"{test_extraction[:50]}...\"")


        filename = os.path.basename(pdf_path)
        book_title = title or os.path.splitext(filename)[0]
        
        logger.info(f"Processing book: {book_title} (ID: {book_id})")
        
        # Add detailed logging for debugging
        logger.info(f"Starting text extraction from {pdf_path}")
        
        # Extract text with page tracking
        extracted_data = extract_text_from_pdf(pdf_path)
        
        # Add detailed logging about extraction results
        logger.info(f"Extraction complete - Total pages found: {extracted_data['total_pages']}")
        logger.info(f"Pages with extractable content: {len(extracted_data['pages'])}")
        
        if not extracted_data["pages"]:
            logger.warning(f"No extractable text found in {filename}, creating minimal record")
            return {
                "book_id": book_id,
                "title": book_title,
                "filename": filename,
                "pages": extracted_data["total_pages"],
                "chunks": 0
            }
                
        total_pages = extracted_data["total_pages"]
        points = []
        chunk_count = 0
        
        # Rest of the function remains the same...
        
        # Generate a numeric seed from book_id for point_id generation
        import hashlib
        # Create an integer hash from book_id (will be much more reliable)
        hash_object = hashlib.md5(book_id.encode())
        id_seed = int(hash_object.hexdigest(), 16) % 10000000  # Use modulo to keep it manageable
        
        for page_data in extracted_data["pages"]:
            page_number = page_data["page_number"]
            page_content = page_data["content"]
            
            # Split page text into chunks
            chunks = [page_content[i:i+1200] for i in range(0, len(page_content), 1200)]
            
            for chunk_num, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                try:
                    # Create a unique point_id that doesn't rely on string manipulation
                    # This is much more reliable
                    point_id = id_seed + (page_number * 1000) + chunk_num
                    
                    # Generate embedding
                    embedding = sentence_transformer.encode(chunk)
                    
                    # Create point
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk,
                            "book_id": book_id,
                            "book_title": book_title,
                            "filename": filename,
                            "page_number": page_number,
                            "total_pages": total_pages,
                            "chunk_num": chunk_num,
                            "total_chunks_in_page": len(chunks),
                            "upload_date": datetime.now().isoformat()
                        }
                    ))
                    
                    chunk_count += 1
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {chunk_num} on page {page_number}: {str(chunk_error)}")
        
        if points:
            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                try:
                    qdrant_client.upsert(
                        collection_name=CONFIG["DEFAULT_COLLECTION"],
                        points=batch_points,
                        wait=True
                    )
                    logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
                except Exception as batch_error:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}: {str(batch_error)}")
        
        return {
            "book_id": book_id,
            "title": book_title,
            "filename": filename,
            "pages": total_pages,
            "chunks": chunk_count
        }
    
    except Exception as e:
        logger.error(f"Failed processing book: {str(e)}")
        # Return basic info even if processing failed
        return {
            "book_id": book_id,
            "title": title or os.path.splitext(os.path.basename(pdf_path))[0],
            "filename": os.path.basename(pdf_path),
            "pages": 0,
            "chunks": 0,
            "error": str(e)
        }

# Chat API endpoints
@app.get("/ai")
async def get_chat():
    history = load_chat_history()
    return {
        "messages": [m for m in state["messages"] if m["role"] != "system"],
        "history": history,
        "medical_mode": state["medical_mode"]
    }

import time

def call_ollama_with_retry(messages, temperature=0.3, max_retries=3):
    """Call Ollama with retry logic"""
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="mistral",
                messages=messages,
                temperature=temperature,
                stream=False,
                timeout=60
            )
            return response
        except Exception as e:
            retry_count += 1
            last_error = e
            logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
            time.sleep(2 ** retry_count)  # Exponential backoff
    
    # If we get here, all retries failed
    raise last_error

@app.post("/ai/chat")
async def post_chat(chat: ChatRequest):
    try:
        state["messages"].append({"role": "user", "content": chat.message})
        system_prompt = MEDICAL_PROMPT if state["medical_mode"] else GENERAL_PROMPT
        messages_to_send = [{"role": "system", "content": system_prompt}] + [m for m in state["messages"] if m["role"] != "system"]
        
        if state["medical_mode"]:
            try:
                # Log that we're searching the database
                logger.info(f"Searching for information related to: {chat.message}")
                
                context = search_qdrant(chat.message)
                
                if context and "No relevant information" not in context:
                    logger.info("Found relevant content in the database")
                    
                    # More explicit context instruction
                    context_instruction = f"""
                    IMPORTANT - I've found these specific passages from uploaded medical documents that are relevant to the query.
                    Read them carefully and base your response on this information:
                    
                    {context}
                    
                    IMPORTANT INSTRUCTIONS FOR USING THIS INFORMATION:
                    1. Focus on using information from these sources ONLY
                    2. ALWAYS cite each specific book and page number using the format [Book Title, Page X]
                    3. Synthesize information from multiple sources if they're complementary
                    4. If sources contradict, note the discrepancy and cite both sources
                    5. DO NOT invent or fabricate additional details
                    6. If the information doesn't fully answer the query, acknowledge this limitation
                    7. Make it obvious you're using information from uploaded documents
                    """
                    
                    # Insert at position 1 to give it high priority
                    messages_to_send.insert(1, {"role": "system", "content": context_instruction})
                    
                    # Also include an example of how to format the response with citations
                    citation_example = """
                    Example of proper citation format:
                    
                    "Study findings showed that treatment X reduced symptoms by 45% compared to placebo [Medical Journal Title, Page 157]. Additionally, a long-term follow-up demonstrated sustained benefits over 5 years [Same Book Title, Page 203]."
                    """
                    
                    messages_to_send.insert(2, {"role": "system", "content": citation_example})
                else:
                    logger.info("No relevant content found in database")
            except Exception as search_error:
                logger.error(f"Error searching Qdrant: {str(search_error)}")
                # Continue without context rather than failing completely
        
        try:
            # Use retry logic for more reliability
            response = call_ollama_with_retry(messages_to_send)
            answer = response.choices[0].message.content
            
            # Add a check if citations were included when context was provided
            if state["medical_mode"] and "No relevant information" not in context and "[" not in answer and "]" not in answer:
                # If no citations found, append a reminder
                answer += "\n\n(Note: The information above is drawn from the medical documents in our database, but specific citations were not included in this response.)"
            
            state["messages"].append({"role": "assistant", "content": answer})
            return {"messages": [m for m in state["messages"] if m["role"] != "system"]}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": f"Error generating response: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return {"error": "An unexpected error occurred. Please try again."}

@app.post("/ai/new_session")
async def new_session():
    if len(state["messages"]) > 1:
        save_chat_session(
            session_id=state["current_session_id"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            messages=[m for m in state["messages"] if m["role"] != "system"]
        )
    state["messages"] = [{"role": "system", "content": MEDICAL_PROMPT if state["medical_mode"] else GENERAL_PROMPT}]
    state["current_session_id"] = str(uuid.uuid4())
    return {"messages": [], "history": load_chat_history()}

@app.get("/ai/session/{session_id}")
async def get_session(session_id: str):
    try:
        with open(CONFIG["HISTORY_FILE"], "r") as f:
            history = json.load(f)
        for session in history:
            if session["session_id"] == session_id:
                state["messages"] = [{"role": "system", "content": MEDICAL_PROMPT if state["medical_mode"] else GENERAL_PROMPT}] + session["messages"]
                state["current_session_id"] = session_id
                return {"messages": [m for m in state["messages"] if m["role"] != "system"]}
        return {"messages": []}
    except Exception as e:
        return {"messages": []}

@app.post("/ai/toggle_mode")
async def toggle_mode(toggle: ToggleRequest):
    state["medical_mode"] = toggle.medical_mode
    state["messages"] = [{"role": "system", "content": MEDICAL_PROMPT if state["medical_mode"] else GENERAL_PROMPT}] + [m for m in state["messages"] if m["role"] != "system"]
    return {"medical_mode": state["medical_mode"]}

@app.get("/ai/test-qdrant")
async def test_qdrant():
    try:
        # List all collections
        collections = qdrant_client.get_collections()
        logger.info(f"Qdrant collections: {collections.collections}")
        
        # Count points in our collection
        collection_info = qdrant_client.get_collection(CONFIG["DEFAULT_COLLECTION"])
        logger.info(f"Collection info: {collection_info}")
        
        # Try to insert a test point
        test_id = 9999999
        test_vector = [0.1] * VECTOR_SIZE
        
        qdrant_client.upsert(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            points=[models.PointStruct(
                id=test_id,
                vector=test_vector,
                payload={"text": "Test point"}
            )],
            wait=True
        )
        
        # Try to search for the test point
        results = qdrant_client.search(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            query_vector=test_vector,
            limit=1
        )
        
        # Clean up the test point
        qdrant_client.delete(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            points_selector=models.PointIdsList(points=[test_id])
        )
        
        if results:
            return {"status": "success", "message": "Qdrant is working correctly"}
        else:
            return {"status": "error", "message": "Qdrant search failed to find test point"}
            
    except Exception as e:
        logger.error(f"Qdrant test failed: {str(e)}")
        return {"status": "error", "message": f"Qdrant test failed: {str(e)}"}

# Book API endpoints
@app.post("/ai/books/upload", response_model=BookUploadResponse)
async def upload_book(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """Upload and process a medical book (PDF) with improved large file handling"""
    try:
        # Validate file type and filename
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext != '.pdf':
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        # Check for potentially malicious characters in filename
        if not all(c.isalnum() or c in ['.', '_', '-'] for c in file.filename):
            raise HTTPException(status_code=400, detail="Invalid filename: Filename contains disallowed characters.")

        # Validate PDF file content
        try:
            pdf_header = await file.read(512)  # Read the first 512 bytes
            if not pdf_header.startswith(b'%PDF-'):
                raise HTTPException(status_code=400, detail="Invalid PDF: File header is not a valid PDF header.")
            await file.seek(0)  # Reset file pointer to the beginning
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: Could not validate PDF header. {str(e)}")

        # Generate unique ID for this book
        book_id = str(uuid.uuid4())
        
        # Initialize upload tracking
        upload_tracker.start_upload(book_id)
        
        # Log the upload attempt with file size info
        file_size = 0
        try:
            # Try to get file size (might not be available for all uploads)
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()  # Get position (file size)
            file.file.seek(0)  # Reset to beginning
            logger.info(f"Uploading file: {file.filename}, Size: {file_size/1024/1024:.2f} MB")
            # Update upload tracker with total size
            upload_tracker.update_progress(book_id, 0, file_size)
        except Exception:
            logger.info(f"Uploading file: {file.filename}, Size unknown")
        
        # Create directory for book if it doesn't exist
        book_dir = os.path.join(CONFIG["UPLOAD_FOLDER"], book_id)
        if os.path.exists(book_dir):
            shutil.rmtree(book_dir)  # Remove it if it exists (unlikely but possible)
        os.makedirs(book_dir, exist_ok=True)
        
        # Save file with a more reliable approach for large files
        file_path = os.path.join(book_dir, file.filename)
        
        # Use chunked writing for more efficient large file handling
        try:
            with open(file_path, "wb") as buffer:
                # Write in 1MB chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB
                current_size = 0
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    current_size += len(chunk)
                    # Update progress in the tracker
                    upload_tracker.update_progress(book_id, current_size, file_size)
                    
            logger.info(f"File saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
            # Mark file upload as complete
            upload_tracker.complete_upload(book_id)
        except Exception as save_error:
            logger.error(f"Failed to save file: {str(save_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(save_error)}")
        
        # Check if file was saved correctly
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File was not saved correctly")
            
        # Log file details
        logger.info(f"File saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
        
        # Process book
        try:
            # Update status to show we're now processing the book
            upload_tracker.uploads[book_id]["status"] = "processing"
            
            result = process_book(
                pdf_path=file_path,
                book_id=book_id,
                title=title
            )
            
            # Even if no chunks were processed, return success if we have basic metadata
            status = "success" if result.get("chunks", 0) > 0 else "partial_success"
            
            # Update final status to completed
            upload_tracker.uploads[book_id]["status"] = "completed"
            
            return {
                "status": status,
                "book_id": result["book_id"],
                "title": result["title"],
                "pages": result.get("pages", 0),
                "chunks": result.get("chunks", 0)
            }
        except Exception as process_error:
            # If processing fails but the file was saved, still return partial success
            logger.error(f"Book processing error: {str(process_error)}")
            # Update status to show processing failed
            upload_tracker.uploads[book_id]["status"] = "processing_failed"
            
            return {
                "status": "partial_success",
                "book_id": book_id,
                "title": title or os.path.splitext(file.filename)[0],
                "pages": 0,
                "chunks": 0
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_book: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process book: {str(e)}")

@app.get("/ai/books", response_model=List[BookInfo])
async def list_books():
    """List all books in the system with improved error handling"""
    try:
        # Get all unique book IDs from the collection
        books = {}
        
        try:
            # First try using the scroll method with proper unpacking
            scroll_results = qdrant_client.scroll(
                collection_name=CONFIG["DEFAULT_COLLECTION"],
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            
            # Handle the results correctly
            if isinstance(scroll_results, list):
                for result in scroll_results:
                    # In newer Qdrant client versions, scroll returns (batch, next_page_offset)
                    if isinstance(result, tuple):
                        batch = result[0]
                    batch = result[0]
                else:
                    # Handle case where result might not be unpacked correctly
                    batch = result
                
                for point in batch:
                    book_id = point.payload.get("book_id")
                    if book_id and book_id not in books:
                        books[book_id] = {
                            "book_id": book_id,
                            "title": point.payload.get("book_title", "Unknown"),
                            "filename": point.payload.get("filename", "Unknown"),
                            "pages": point.payload.get("total_pages", 0),
                            "chunks": 0,  # Will count below
                            "upload_date": point.payload.get("upload_date", "Unknown")
                        }
                    
                    # Count chunks for this book
                    if book_id and book_id in books:
                        books[book_id]["chunks"] += 1
                        
        except Exception as scroll_error:
            logger.error(f"Error using scroll method: {str(scroll_error)}")
            
            # Fallback to search method if scroll fails
            logger.info("Falling back to search method to list books")
            search_result = qdrant_client.search(
                collection_name=CONFIG["DEFAULT_COLLECTION"],
                query_vector=[0.0] * VECTOR_SIZE,  # Dummy vector
                limit=1000,  # Get more results
                with_payload=True,
                with_vectors=False,
                score_threshold=0.0  # Get all records
            )
            
            for point in search_result:
                book_id = point.payload.get("book_id")
                if book_id and book_id not in books:
                    books[book_id] = {
                        "book_id": book_id,
                        "title": point.payload.get("book_title", "Unknown"),
                        "filename": point.payload.get("filename", "Unknown"),
                        "pages": point.payload.get("total_pages", 0),
                        "chunks": 0,
                        "upload_date": point.payload.get("upload_date", "Unknown")
                    }
                
                # Count chunks for this book
                if book_id and book_id in books:
                    books[book_id]["chunks"] += 1
        
        return list(books.values())
        
    except Exception as e:
        logger.error(f"Error listing books: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")

@app.put("/ai/books/{book_id}")
async def update_book(book_id: str, update_request: BookUpdateRequest):
    """Update book metadata such as title"""
    try:
        # Verify book exists
        query_result = qdrant_client.scroll(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_id",
                        match=models.MatchValue(value=book_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        
        batch, _ = next(query_result, ([], None))
        if not batch:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        
        # Update the payload for all chunks of this book
        update_payload = {}
        if update_request.title:
            update_payload["book_title"] = update_request.title
        if update_request.author:
            update_payload["author"] = update_request.author
            
        if not update_payload:
            raise HTTPException(status_code=400, detail="No updates provided")
            
        qdrant_client.set_payload(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            payload=update_payload,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_id",
                        match=models.MatchValue(value=book_id),
                    )
                ]
            ),
            wait=True
        )
        
        return {"status": "success", "message": f"Book {book_id} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating book: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update book: {str(e)}")

@app.delete("/ai/books/{book_id}")
async def delete_book(book_id: str):
    """Delete a book from the system"""
    try:
        # Find all points with this book_id
        query_result = qdrant_client.scroll(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_id",
                        match=models.MatchValue(value=book_id),
                    )
                ]
            ),
            limit=100,
            with_payload=False,
            with_vectors=False,
        )
        
        # Collect all point IDs
        point_ids = []
        for batch, _ in query_result:
            point_ids.extend([point.id for point in batch])
        
        if not point_ids:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        
        # Delete points in batches
        batch_size = 100
        for i in range(0, len(point_ids), batch_size):
            qdrant_client.delete(
                collection_name=CONFIG["DEFAULT_COLLECTION"],
                points_selector=models.PointIdsList(
                    points=point_ids[i:i+batch_size]
                ),
                wait=True
            )
        
        # Delete file if it exists
        for filename in os.listdir(CONFIG["UPLOAD_FOLDER"]):
            if filename.startswith(f"{book_id}_"):
                os.remove(os.path.join(CONFIG["UPLOAD_FOLDER"], filename))
                break
        
        return {"status": "success", "message": f"Book {book_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting book: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")

@app.post("/ai/search")
async def search_books(
    request: SearchRequest,
    book_id: Optional[str] = Query(None, description="Filter by specific book ID")
):
    """Search across all books or a specific book"""
    try:
        # Generate query embedding
        query_vector = sentence_transformer.encode(request.query).tolist()
        
        # Prepare filters if book_id is specified
        search_filter = None
        if book_id:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_id",
                        match=models.MatchValue(value=book_id),
                    )
                ]
            )
        
        # Search in collection
        search_result = qdrant_client.search(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            query_vector=query_vector,
            query_filter=search_filter,
            limit=request.limit,
            with_payload=True,
            score_threshold=request.threshold
        )
        
        # Format results with source attribution
        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "book_title": hit.payload.get("book_title", "Unknown"),
                "book_id": hit.payload.get("book_id", "Unknown"),
                "page_number": hit.payload.get("page_number", 0),
                "total_pages": hit.payload.get("total_pages", 0),
                "relevance_score": hit.score
            })
        
        return {
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching books: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/debug/routes")
def get_routes():
    """List all registered routes for debugging"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods) if route.methods else []
        })
    return {"routes": routes}

@app.get("/ai/health")
async def health_check():
    """Check if API and Qdrant are healthy"""
    try:
        qdrant_client.get_collections()
        return {"status": "healthy", "qdrant_connected": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}



if __name__ == "__main__":
    import uvicorn
    
    # Configure Uvicorn with larger limits
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # Increase timeout for large uploads
        timeout_keep_alive=300,  # 5 minutes in seconds
        # Allow large requests
        limit_concurrency=10,
        # Buffer size for large uploads
        limit_max_requests=10
    )
