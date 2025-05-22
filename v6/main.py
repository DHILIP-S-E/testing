from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from typing import List, Optional, Dict, Any
import uvicorn
import openai
import uuid
import json
import os
import re
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import shutil
import logging
from openai import APIError, APIConnectionError, APITimeoutError
from starlette.datastructures import UploadFile as StarletteUploadFile
import time
import hashlib
import subprocess
import tempfile
import requests
from exam import generate_mcqs_with_mistral, create_default_mcq_options, save_exam_to_file

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
    "OLLAMA_API_URL": os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate"),
    "EXAMS_FOLDER": "./generated_exams",
    "MAX_CHAT_HISTORY_FOR_MODEL": 20, # Max number of user/assistant messages to send to the model (10 pairs)
}

# Create directories
os.makedirs(CONFIG["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(CONFIG["EXAMS_FOLDER"], exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI System"
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("chat.html", "r") as f:
        return HTMLResponse(content=f.read())

# Add a custom middleware to handle large file uploads
@app.middleware("http")
async def add_custom_header(request: Request, call_next):
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

# Initialize clients and models
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

qdrant_client = QdrantClient(
    url=CONFIG["QDRANT_URL"].replace("http://", "https://") if CONFIG["QDRANT_URL"].startswith("http://") else CONFIG["QDRANT_URL"],
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

CONVERSATIONAL GUIDELINES:
- Maintain a conversational and helpful tone.
- If a user repeats a question, and the core information from the provided sources for that query has already been presented, explicitly state that you have provided the available information on that topic. Then, offer to rephrase it, provide a summary, or suggest exploring a related but different aspect of the topic, or a new topic entirely. Avoid simply repeating the exact same response.
- Encourage further questions or deeper exploration of the topic if appropriate.
- Acknowledge previous turns to maintain continuity, e.g., "Building on our previous discussion about X..."
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
    "medical_mode": False,
    "medi_ai_mode": False # New toggle for MediAI mode (Qdrant-based)
}

# Request/response models
class ChatRequest(BaseModel):
    message: str

class ToggleRequest(BaseModel):
    medi_ai_mode: bool

class CombinedChatResponse(BaseModel):
    medi_ai_response: str
    common_response: str
    messages: List[Dict[str, Any]] # To keep track of chat history

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

class MCQOption(BaseModel):
    option: str
    text: str
    correct: bool

class MCQQuestion(BaseModel):
    question: str
    options: List[MCQOption]
    explanation: str
    marks: int

class ExamRequest(BaseModel):
    exam_name: str
    book_name: str
    date: str
    duration: str
    total_questions: int
    marks_per_question: int
    ai_option: str = "MediAI"
    collection_name: Optional[str] = None

class ExamResponse(BaseModel):
    exam_id: str
    exam_name: str
    book_name: str
    date: str
    duration: str
    total_questions: int
    marks_per_question: int
    total_marks: int
    questions: List[MCQQuestion]

# Helper functions
def ensure_history_file():
    if not os.path.exists(CONFIG["HISTORY_FILE"]):
        with open(CONFIG["HISTORY_FILE"], "w") as f:
            json.dump([], f)



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

def search_qdrant(query, collection_name: str = CONFIG["DEFAULT_COLLECTION"]):
    try:
        # Log the search query for debugging
        logger.info(f"Searching for: {query} in collection: {collection_name}")
        
        # Try multiple search approaches
        logger.info("Generating embedding for query")
        start_time = time.time()
        query_vector = get_embedding(query)
        end_time = time.time()
        logger.info(f"Embedding generated in {end_time - start_time:.2f} seconds")
        # First search with higher threshold for precision
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=16,
            with_payload=True,
            score_threshold=0.5  # Higher threshold for better quality
        )

        # If no results, try again with lower threshold
        if not search_result:
            logger.info("No results with high threshold, trying lower threshold")
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=16,
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

def query_qdrant_for_book(book_name: str, collection_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Query Qdrant for chunks belonging to a specific book within a collection."""
    try:
        logger.info(f"Querying Qdrant for book '{book_name}' in collection '{collection_name}' with limit {limit}")

        # First, find the book_id for the given book_name
        # This assumes book_name is unique enough or we need to search by title
        # For simplicity, let's assume book_name directly maps to book_title in payload
        
        # We need a dummy vector for scroll if we don't have a specific query
        # Or, we can use a filter directly with scroll
        
        # Use scroll with a filter to get points for the specific book
        scroll_results = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_title",
                        match=models.MatchValue(value=book_name),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        book_chunks = []
        for batch, _ in scroll_results:
            for point in batch:
                book_chunks.append(point.payload)
        
        logger.info(f"Found {len(book_chunks)} chunks for book '{book_name}' in collection '{collection_name}'")
        return book_chunks

    except Exception as e:
        logger.error(f"Error querying Qdrant for book '{book_name}': {str(e)}")
        return []

def retry_operation(operation, max_retries=3):
    """Retry an operation with exponential backoff"""
    retries = 0
    last_error = None
    
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
    
    # If we get here, all retries failed
    raise last_error

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
                    logger.debug(f"Standard extraction failed for page {page_num+1}, trying alternative methods")
                    
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
        
        if not pages_content and reader.pages:
            logger.warning(f"Standard text extraction failed for {pdf_path}, trying more aggressive methods")
            try:
                logger.info("Attempting extraction with external tools")
                
                with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                    temp_path = temp.name
                
                try:
                    subprocess.run(['pdftotext', pdf_path, temp_path], 
                                           check=True, capture_output=True)
                    
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
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

def process_book(pdf_path: str, book_id: str, title: str = None) -> dict:
    """Process a single book and add to vector database with robust error handling"""
    try:
        test_extraction = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages[:3]:
                    text = page.extract_text()
                    if text and len(text) > 10:
                        test_extraction = text[:200]
                        break
        except Exception as ex:
            logger.error(f"Basic PDF test extraction failed: {str(ex)}")
            
        if not test_extraction:
            logger.error(f"❌ CRITICAL: Cannot extract any text from {pdf_path}")
            logger.error("This is likely why your collection is empty")
            
            try:
                file_size = os.path.getsize(pdf_path)
                with open(pdf_path, 'rb') as f:
                    header = f.read(1024)
                logger.error(f"File size: {file_size} bytes, Header starts with: {header[:20]}")
            except Exception as ex:
                logger.error(f"File analysis failed: {str(ex)}")
                
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
        
        logger.info(f"Starting text extraction from {pdf_path}")
        
        extracted_data = extract_text_from_pdf(pdf_path)
        
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
        
        hash_object = hashlib.md5(book_id.encode())
        id_seed = int(hash_object.hexdigest(), 16) % 10000000
        
        for page_data in extracted_data["pages"]:
            page_number = page_data["page_number"]
            page_content = page_data["content"]
            
            chunks = [page_content[i:i+1200] for i in range(0, len(page_content), 1200)]
            
            for chunk_num, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                try:
                    point_id = id_seed + (page_number * 1000) + chunk_num
                    
                    embedding = sentence_transformer.encode(chunk)
                    
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
        "medical_mode": state["medical_mode"],
        "medi_ai_mode": state["medi_ai_mode"] # Return new toggle state
    }

def call_ollama_with_retry(messages, temperature=0.7, max_retries=3):
    """Call Ollama with retry logic"""
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            logger.info("Calling Ollama API")
            start_time = time.time()
            response = client.chat.completions.create(
                model="mistral",
                messages=messages,
                temperature=temperature,
                stream=False,
                timeout=120,
            )
            end_time = time.time()
            logger.info(f"Ollama API call completed in {end_time - start_time:.2f} seconds")
            return response
        except Exception as e:
            retry_count += 1
            last_error = e
            logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
            time.sleep(2 ** retry_count)
    
    raise last_error

@app.post("/ai/chat", response_model=CombinedChatResponse)
async def post_chat(chat: ChatRequest):
    try:
        user_message = chat.message
        state["messages"].append({"role": "user", "content": user_message})
        
        # Filter out the system message and get the actual conversation history
        conversation_history = [m for m in state["messages"] if m["role"] != "system"]
        
        # Truncate conversation history to MAX_CHAT_HISTORY_FOR_MODEL messages
        truncated_history = conversation_history[-CONFIG["MAX_CHAT_HISTORY_FOR_MODEL"]:]

        # --- Medi AI Mode (Qdrant-based only, formatted by LLM if context exists) ---
        medi_ai_final_answer = ""
        try:
            search_query_medi_ai = user_message
            context_medi_ai = search_qdrant(search_query_medi_ai, collection_name=CONFIG["DEFAULT_COLLECTION"])
            
            if context_medi_ai and "No relevant information" not in context_medi_ai:
                system_prompt_medi_ai = MEDICAL_PROMPT
                messages_to_send_medi_ai = [{"role": "system", "content": system_prompt_medi_ai}] + truncated_history
                
                context_instruction_medi_ai = f"""
                IMPORTANT - I've found these specific passages from the medical documents in the Qdrant vector database that are relevant to the query.
                Read them carefully and base your response *strictly and exclusively* on this information.
                
                {context_medi_ai}
                
                IMPORTANT INSTRUCTIONS FOR USING THIS INFORMATION:
                1. Your response MUST be derived *only* from the provided sources. DO NOT use any external knowledge.
                2. ALWAYS cite each specific book and page number using the format [Book Title, Page X].
                3. Synthesize information from multiple sources if they're complementary.
                4. If sources contradict, note the discrepancy and cite both sources.
                5. DO NOT invent or fabricate additional details or make claims without supporting evidence from the sources.
                6. If the provided information doesn't fully answer the query, acknowledge this limitation clearly.
                7. Make it obvious you're using information from the Qdrant vector database.
                """
                messages_to_send_medi_ai.insert(1, {"role": "system", "content": context_instruction_medi_ai})
                citation_example_medi_ai = """
                Example of proper citation format:
                
                "Study findings showed that treatment X reduced symptoms by 45% compared to placebo [Book Title, Page 157]. Additionally, a long-term follow-up demonstrated sustained benefits over 5 years [Another Book Title, Page 203]."
                """
                messages_to_send_medi_ai.insert(2, {"role": "system", "content": citation_example_medi_ai})
                
                response_medi_ai = call_ollama_with_retry(messages_to_send_medi_ai)
                medi_ai_final_answer = response_medi_ai.choices[0].message.content
                
                if "No relevant information" not in context_medi_ai and "[" not in medi_ai_final_answer and "]" not in medi_ai_final_answer:
                    medi_ai_final_answer += "\n\n(Note: The information above is drawn from the medical documents in our Qdrant database, but specific citations were not included in this response.)"
            else:
                logger.info("No relevant content found in Qdrant for Medi AI, providing direct 'no info' message.")
                medi_ai_final_answer = "Medi AI (Retrieval-Based): I am unable to find relevant information in the medical eBook sources for your query. Please try rephrasing your question or asking about a different medical topic."
        except Exception as e:
            logger.error(f"Error generating Medi AI response: {str(e)}")
            medi_ai_final_answer = f"Medi AI (Retrieval-Based): Error generating response: {str(e)}"

        # --- Common Mode (Hybrid: Qdrant + Mistral) ---
        common_final_answer = ""
        try:
            system_prompt_common = MEDICAL_PROMPT # Common mode still uses medical books
            messages_to_send_common = [{"role": "system", "content": system_prompt_common}] + truncated_history
            
            context_common = search_qdrant(user_message, collection_name=CONFIG["DEFAULT_COLLECTION"])
            
            if context_common and "No relevant information" not in context_common:
                context_instruction_common = f"""
                IMPORTANT - I've found these specific passages from the medical documents in the Qdrant vector database that are relevant to the query.
                Read them carefully and base your response on this information:
                
                {context_common}
                
                IMPORTANT INSTRUCTIONS FOR USING THIS INFORMATION:
                1. Focus on using information from these sources ONLY
                2. ALWAYS cite each specific book and page number using the format [Book Title, Page X]
                3. Synthesize information from multiple sources if they're complementary.
                4. If sources contradict, note the discrepancy and cite both sources.
                5. DO NOT invent or fabricate additional details
                6. If the information doesn't fully answer the query, acknowledge this limitation
                7. Make it obvious you're using information from the Qdrant vector database
                """
                messages_to_send_common.insert(1, {"role": "system", "content": context_instruction_common})
                citation_example_common = """
                Example of proper citation format:
                
                "Study findings showed that treatment X reduced symptoms by 45% compared to placebo [Book Title, Page 157]. Additionally, a long-term follow-up demonstrated sustained benefits over 5 years [Another Book Title, Page 203]."
                """
                messages_to_send_common.insert(2, {"role": "system", "content": citation_example_common})
            else:
                logger.info("No relevant content found in Qdrant for Common mode, proceeding with general knowledge.")
                # If no context, the model will rely on its general knowledge based on MEDICAL_PROMPT
            
            response_common = call_ollama_with_retry(messages_to_send_common)
            common_final_answer = response_common.choices[0].message.content
            
            if "No relevant information" not in context_common and "[" not in common_final_answer and "]" not in common_final_answer:
                common_final_answer += "\n\n(Note: The information above is drawn from the medical documents in our Qdrant database, but specific citations were not included in this response.)"

        except Exception as e:
            logger.error(f"Error generating Common mode response: {str(e)}")
            common_final_answer = f"Common Mode: Error generating response: {str(e)}"

        # Append the common mode's assistant response to the state messages for context remembrance
        # This assumes the common mode's response is the primary conversational flow.
        state["messages"].append({"role": "assistant", "content": common_final_answer})

        return CombinedChatResponse(
            medi_ai_response=medi_ai_final_answer,
            common_response=common_final_answer,
            messages=[m for m in state["messages"] if m["role"] != "system"]
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")

@app.post("/ai/new_session")
async def new_session():
    if len(state["messages"]) > 1:
        save_chat_session(
            session_id=state["current_session_id"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            messages=[m for m in state["messages"] if m["role"] != "system"]
        )
    # Save current session before starting a new one
    if len(state["messages"]) > 1:
        save_chat_session(
            session_id=state["current_session_id"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            messages=[m for m in state["messages"] if m["role"] != "system"]
        )
    
    # Start a truly new session with a new UUID
    new_session_id = str(uuid.uuid4())
    state["current_session_id"] = new_session_id
    state["messages"] = [{"role": "system", "content": MEDICAL_PROMPT if state["medical_mode"] else GENERAL_PROMPT}]
    
    return {
        "messages": [m for m in state["messages"] if m["role"] != "system"],
        "history": [], # Ensure history is empty for new sessions
        "medical_mode": state["medical_mode"],
        "medi_ai_mode": state["medi_ai_mode"]
    }

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
    state["medi_ai_mode"] = toggle.medi_ai_mode
    # Implicitly set medical_mode based on medi_ai_mode
    state["medical_mode"] = toggle.medi_ai_mode # True if medi_ai_mode is True, False otherwise
        
    # Adjust system prompt based on medical_mode
    state["messages"] = [{"role": "system", "content": MEDICAL_PROMPT if state["medical_mode"] else GENERAL_PROMPT}] + [m for m in state["messages"] if m["role"] != "system"]
    
    return {"medi_ai_mode": state["medi_ai_mode"]}

@app.get("/ai/test-qdrant")
async def test_qdrant():
    try:
        collections = qdrant_client.get_collections()
        logger.info(f"Qdrant collections: {collections.collections}")
        
        collection_info = qdrant_client.get_collection(CONFIG["DEFAULT_COLLECTION"])
        logger.info(f"Collection info: {collection_info}")
        
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
        
        results = qdrant_client.search(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            query_vector=test_vector,
            limit=1
        )
        
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
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext != '.pdf':
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        if not all(c.isalnum() or c in ['.', '_', '-'] for c in file.filename):
            raise HTTPException(status_code=400, detail="Invalid filename: Filename contains disallowed characters.")

        try:
            pdf_header = await file.read(512)
            if not pdf_header.startswith(b'%PDF-'):
                raise HTTPException(status_code=400, detail="Invalid PDF: File header is not a valid PDF header.")
            await file.seek(0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: Could not validate PDF header. {str(e)}")

        book_id = str(uuid.uuid4())
        
        upload_tracker.start_upload(book_id)
        
        file_size = 0
        try:
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)
            logger.info(f"Uploading file: {file.filename}, Size: {file_size/1024/1024:.2f} MB")
            upload_tracker.update_progress(book_id, 0, file_size)
        except Exception:
            logger.info(f"Uploading file: {file.filename}, Size unknown")
        
        book_dir = os.path.join(CONFIG["UPLOAD_FOLDER"], book_id)
        if os.path.exists(book_dir):
            shutil.rmtree(book_dir)
        os.makedirs(book_dir, exist_ok=True)
        
        file_path = os.path.join(book_dir, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                chunk_size = 1024 * 1024
                current_size = 0
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    current_size += len(chunk)
                    upload_tracker.update_progress(book_id, current_size, file_size)
                    
            logger.info(f"File saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
            upload_tracker.complete_upload(book_id)
        except Exception as save_error:
            logger.error(f"Failed to save file: {str(save_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(save_error)}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File was not saved correctly")
            
        logger.info(f"File saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
        
        try:
            upload_tracker.uploads[book_id]["status"] = "processing"
            
            result = process_book(
                pdf_path=file_path,
                book_id=book_id,
                title=title
            )
            
            status = "success" if result.get("chunks", 0) > 0 else "partial_success"
            
            upload_tracker.uploads[book_id]["status"] = "completed"
            
            return {
                "status": status,
                "book_id": result["book_id"],
                "title": result["title"],
                "pages": result.get("pages", 0),
                "chunks": result.get("chunks", 0)
            }
        except Exception as process_error:
            logger.error(f"Book processing error: {str(process_error)}")
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
        books = {}
        
        try:
            scroll_results = qdrant_client.scroll(
                collection_name=CONFIG["DEFAULT_COLLECTION"],
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            
            all_points = []
            for batch, _ in scroll_results:
                all_points.extend(batch)
            
            for point in all_points:
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
                
                if book_id and book_id in books:
                    books[book_id]["chunks"] += 1
                        
        except Exception as scroll_error:
            logger.error(f"Error using scroll method: {str(scroll_error)}")
            
            logger.info("Falling back to search method to list books")
            search_result = qdrant_client.search(
                collection_name=CONFIG["DEFAULT_COLLECTION"],
                query_vector=[0.0] * VECTOR_SIZE,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                score_threshold=0.0
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
        
        point_ids = []
        for batch, _ in query_result:
            point_ids.extend([point.id for point in batch])
        
        if not point_ids:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        
        batch_size = 100
        for i in range(0, len(point_ids), batch_size):
            qdrant_client.delete(
                collection_name=CONFIG["DEFAULT_COLLECTION"],
                points_selector=models.PointIdsList(
                    points=point_ids[i:i+batch_size]
                ),
                wait=True
            )
        
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
        query_vector = sentence_transformer.encode(request.query).tolist()
        
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
        
        search_result = qdrant_client.search(
            collection_name=CONFIG["DEFAULT_COLLECTION"],
            query_vector=query_vector,
            query_filter=search_filter,
            limit=request.limit,
            with_payload=True,
            score_threshold=request.threshold
        )
        
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

# Exam API endpoints
@app.post("/ai/generate-questions", response_model=ExamResponse)
async def generate_exam(exam_request: ExamRequest):
    logger.info(f"Generating exam: {exam_request.exam_name} for book: {exam_request.book_name}, AI option: {exam_request.ai_option}")

    is_common_ai = exam_request.ai_option.lower() == "common ai"

    if not is_common_ai and (not sentence_transformer or not qdrant_client):
        raise HTTPException(
            status_code=500,
            detail="Embedder or Qdrant client not initialized. Check server logs."
        )

    if is_common_ai:
        logger.info(f"Using Common AI to generate questions based on input: {exam_request.book_name}")
        context = exam_request.book_name
    else:
        logger.info(f"Using Medi AI to generate questions from Qdrant books")

        if not exam_request.collection_name:
            logger.info(f"No collection name provided, using book_name directly as context")
            context = exam_request.book_name
        else:
            book_chunks = query_qdrant_for_book(
                book_name=exam_request.book_name,
                collection_name=exam_request.collection_name,
                limit=50,
            )

            if not book_chunks:
                raise HTTPException(
                    status_code=404,
                    detail=f"No content found for book '{exam_request.book_name}' in collection '{exam_request.collection_name}'."
                )

            context = "\n\n".join([chunk["text"] for chunk in book_chunks])
            logger.info(f"Created context with {len(context)} characters from {len(book_chunks)} chunks")

    mcq_questions = generate_mcqs_with_mistral(context, exam_request, is_common_ai)

    max_retries = 3
    retry_count = 0

    while len(mcq_questions) < exam_request.total_questions and retry_count < max_retries:
        retry_count += 1
        logger.info(f"Got only {len(mcq_questions)} unique questions, trying again (attempt {retry_count}/{max_retries})")

        retry_request = ExamRequest(
            exam_name=exam_request.exam_name,
            book_name=exam_request.book_name,
            date=exam_request.date,
            duration=exam_request.duration,
            total_questions=exam_request.total_questions * (retry_count + 1),
            marks_per_question=exam_request.marks_per_question,
            ai_option=exam_request.ai_option,
            collection_name=exam_request.collection_name
        )

        new_questions = generate_mcqs_with_mistral(context, retry_request, is_common_ai)

        existing_question_texts = {q.get("question", "").strip().lower() for q in mcq_questions}
        for q in new_questions:
            question_text = q.get("question", "").strip().lower()
            if question_text and question_text not in existing_question_texts:
                mcq_questions.append(q)
                existing_question_texts.add(question_text)

        if len(mcq_questions) >= exam_request.total_questions:
            break

    if not mcq_questions:
        raise HTTPException(
            status_code=500,
            detail="MCQ generation failed. Check server logs for details."
        )

    exam_id = f"exam_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    questions = []
    for q in mcq_questions:
        try:
            if "options" not in q:
                if "answers" in q and "correct_answer" in q:
                    q["options"] = create_default_mcq_options(q["answers"], q["correct_answer"])
                else:
                    logger.warning(f"Question missing options and cannot be converted: {q}")
                    continue

            options = []
            for opt in q.get("options", []):
                if not isinstance(opt, dict) or "option" not in opt or "text" not in opt:
                    continue
                options.append(MCQOption(
                    option=opt.get("option", ""),
                    text=opt.get("text", ""),
                    correct=opt.get("correct", False)
                ))

            if not options:
                logger.warning(f"Question has no valid options: {q}")
                continue

            while len(options) < 4:
                option_letter = chr(65 + len(options))
                options.append(MCQOption(
                    option=option_letter,
                    text=f"Option {option_letter} (placeholder)",
                    correct=False
                ))

            questions.append(MCQQuestion(
                question=q.get("question", ""),
                options=options,
                explanation=q.get("explanation", "See the correct answer marked above."),
                marks=q.get("marks", exam_request.marks_per_question),
            ))
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            continue

    if not questions:
        raise HTTPException(
            status_code=500,
            detail="Failed to create valid questions from generated content."
        )

    if len(questions) > exam_request.total_questions:
        logger.info(f"Limiting from {len(questions)} to {exam_request.total_questions} questions as requested")
        questions = questions[:exam_request.total_questions]

    exam_response = ExamResponse(
        exam_id=exam_id,
        exam_name=exam_request.exam_name,
        book_name=exam_request.book_name,
        date=exam_request.date,
        duration=exam_request.duration,
        total_questions=len(questions),
        marks_per_question=exam_request.marks_per_question,
        total_marks=len(questions) * exam_request.marks_per_question,
        questions=questions,
    )

    save_exam_to_file(exam_response)
    logger.info(f"Successfully generated exam with {len(questions)} questions")
    return exam_response


@app.get("/ai/list-exam-books/{collection_name}")
async def list_exam_books(collection_name: str):
    if not qdrant_client:
        raise HTTPException(
            status_code=500,
            detail="Qdrant client not initialized. Check server logs."
        )

    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist"
            )

        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = scroll_result

        books = set()
        for point in points:
            book_title = point.payload.get("book_title")
            if book_title:
                books.add(book_title)

        logger.info(f"Found {len(books)} books in collection {collection_name}")
        return {"books": sorted(list(books))}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List books error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")


@app.get("/ai/collections")
async def list_collections():
    if not qdrant_client:
        raise HTTPException(
            status_code=500,
            detail="Qdrant client not initialized. Check server logs."
        )

    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        logger.info(f"Found {len(collection_names)} collections")
        return {"collections": collection_names}
    except Exception as e:
        logger.error(f"List collections error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.get("/ai/exams")
async def list_exams():
    try:
        files = os.listdir(CONFIG["EXAMS_FOLDER"])
        exams = []
        for fname in files:
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(CONFIG["EXAMS_FOLDER"], fname), "r") as f:
                        data = json.load(f)
                        exams.append({
                            "exam_id": data.get("exam_id", ""),
                            "exam_name": data.get("exam_name", ""),
                            "date": data.get("date", ""),
                            "total_questions": data.get("total_questions", 0),
                        })
                except Exception as e:
                    logger.error(f"Error reading exam file {fname}: {e}")

        logger.info(f"Found {len(exams)} exams")
        return {"exams": exams}
    except Exception as e:
        logger.error(f"List exams error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list exams: {str(e)}")


@app.get("/ai/exam/{exam_id}")
async def get_exam(exam_id: str):
    try:
        for fname in os.listdir(CONFIG["EXAMS_FOLDER"]):
            if fname.startswith(exam_id) and fname.endswith(".json"):
                try:
                    with open(os.path.join(CONFIG["EXAMS_FOLDER"], fname), "r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error reading exam file {fname}: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to read exam file: {str(e)}"
                    )

        raise HTTPException(status_code=404, detail=f"Exam with ID '{exam_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get exam error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get exam: {str(e)}")

@app.get("/ai/exam-health")
async def exam_health_check():
    health = {
        "status": "ok" if sentence_transformer and qdrant_client else "error",
        "embedder": "initialized" if sentence_transformer else "error",
        "qdrant": "connected" if qdrant_client else "error",
        "timestamp": datetime.now().isoformat()
    }
    return health

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7000,
        timeout_keep_alive=300
    )
