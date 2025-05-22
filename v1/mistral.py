from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mistral Chat API")

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
QDRANT_URL = "https://b2047c45-446d-46f1-93fd-2739ee654557.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.i577Hj5NyfcWKn06fy_JyvvS6kfN7z4-v4p7_e1b95s"
QDRANT_COLLECTION_NAME = "pdf_documents"

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    logger.info("Successfully connected to Qdrant")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    raise

# In-memory chat history storage (replace with Redis in production)
chat_sessions = {}

# Define request body structure
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # To track conversation history
    max_length: int = 100
    use_mistral_only: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: str
    sources: List[str]
    history_length: int

class DocumentUpload(BaseModel):
    file_path: str
    use_mistral_only: bool = False

# Helper functions
def cosine_similarity(a: List[float], b: List[float]) -> float:
    try:
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

def query_qdrant(query_embedding: List[float], limit: int = 3) -> List[Dict]:
    try:
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return [{
            "id": hit.id,
            "text": hit.payload.get("text", ""),
            "similarity": hit.score
        } for hit in search_result]
    except Exception as e:
        logger.error(f"Qdrant query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Qdrant query error: {str(e)}")

# Chat endpoint with memory
@app.post("/chat", response_model=ChatResponse)
async def chat_with_memory(request: ChatRequest) -> ChatResponse:
    try:
        # Initialize or retrieve chat session
        if not request.session_id or request.session_id not in chat_sessions:
            session_id = str(uuid4())
            chat_sessions[session_id] = []
        else:
            session_id = request.session_id

        # Get previous messages
        chat_history = chat_sessions[session_id]

        # Generate query embedding (if not Mistral-only)
        if not request.use_mistral_only:
            try:
                query_embedding = embedder.encode(request.message).tolist()
                top_docs = query_qdrant(query_embedding)
                context = "\n".join([doc["text"] for doc in top_docs])
            except Exception as e:
                logger.error(f"Error during context retrieval: {str(e)}")
                context = "Error retrieving context"
                top_docs = []
        else:
            context = "No context (Mistral-only mode)"
            top_docs = []

        # Format prompt with history
        history_str = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                               for msg in chat_history[-3:]])  # Keep last 3 exchanges

        prompt = f"""
        Previous conversation:
        {history_str}

        Context from documents:
        {context}

        New question: {request.message}
        Answer:
        """

        # Call Mistral
        try:
            payload = {
                "model": "mistral",
                "prompt": prompt,
                "max_tokens": request.max_length,
                "temperature": 0.7,
                "stream": False
            }
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()

            # Get response and update history
            result = response.json()
            generated_text = result.get("response", "No response generated")

            chat_sessions[session_id].append({
                "user": request.message,
                "assistant": generated_text
            })

            return ChatResponse(
                response=generated_text,
                session_id=session_id,
                context_used=context,
                sources=[doc["id"] for doc in top_docs],
                history_length=len(chat_sessions[session_id])
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Mistral API: {str(e)}")
            raise HTTPException(status_code=503, detail="Failed to generate response from Mistral")

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Upload document endpoint (to be implemented)
@app.post("/upload-document")
async def upload_document(request: DocumentUpload):
    """
    Endpoint for uploading documents to be processed and stored in Qdrant.
    This is a placeholder for future implementation.
    """
    raise HTTPException(status_code=501, detail="Document upload not implemented yet")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)