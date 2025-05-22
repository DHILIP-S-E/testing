from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import os
from datetime import datetime

# Setup logger
logger = logging.getLogger("exam_generator")
logging.basicConfig(level=logging.INFO)

# Configuration
CONFIG = {
    "QDRANT_URL": "http://127.0.0.1:6333",
    "API_KEY": "",
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "EXAMS_FOLDER": "./generated_exams",
}

# Create exams folder if missing
os.makedirs(CONFIG["EXAMS_FOLDER"], exist_ok=True)

# Load embedding model and Qdrant client
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url=CONFIG["QDRANT_URL"], api_key=CONFIG["API_KEY"])

# Router
prefix_router = APIRouter()

# ----------- Models -----------

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
    collection_name: str

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

# ----------- Helpers -----------

def query_qdrant_for_book(book_name: str, collection_name: str, limit: int = 20):
    try:
        query_embedding = embedder.encode(book_name).tolist()
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_title",
                        match=models.MatchValue(value=book_name)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("book_title", ""),
                "similarity": hit.score,
            }
            for hit in search_result
        ]
    except Exception as e:
        logger.error(f"Qdrant query error: {e}")
        return []

def generate_mcqs_with_mistral(context: str, exam_request: ExamRequest):
    try:
        prompt = f"""
You are a medical exam AI. Generate {exam_request.total_questions} MCQs based on the text below.
Each question should have 4 options (A-D), one correct answer, explanation, and be worth {exam_request.marks_per_question} marks.
Return a JSON array of objects like:
[
  {{
    "question": "...",
    "options": [
      {{"option": "A", "text": "...", "correct": false}},
      {{"option": "B", "text": "...", "correct": true}},
      {{"option": "C", "text": "...", "correct": false}},
      {{"option": "D", "text": "...", "correct": false}}
    ],
    "explanation": "...",
    "marks": {exam_request.marks_per_question}
  }},
  ...
]
Text:
{context}
Return ONLY the JSON array, no extra text.
"""
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "max_tokens": 4000,
            "temperature": 0.3,
            "stream": False,
        }
        resp = requests.post(CONFIG["OLLAMA_API_URL"], json=payload)
        resp.raise_for_status()
        response_text = resp.json().get("response", "")
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        mcq_json = json.loads(response_text[start:end])
        return mcq_json
    except Exception as e:
        logger.error(f"Mistral generation error: {e}")
        return []

def save_exam_to_file(exam_data: ExamResponse):
    try:
        filename = f"{exam_data.exam_id}_{exam_data.exam_name.replace(' ', '_')}.json"
        filepath = os.path.join(CONFIG["EXAMS_FOLDER"], filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(exam_data.dict(), f, indent=2)
        logger.info(f"Saved exam to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save exam: {e}")
        return None

# ----------- Routes -----------

@prefix_router.post("/generate-questions", response_model=ExamResponse)
async def generate_exam(exam_request: ExamRequest):
    book_chunks = query_qdrant_for_book(
        book_name=exam_request.book_name,
        collection_name=exam_request.collection_name,
        limit=50,
    )
    if not book_chunks:
        raise HTTPException(status_code=404, detail="No book content found.")

    context = "\n\n".join([chunk["text"] for chunk in book_chunks])
    mcq_questions = generate_mcqs_with_mistral(context, exam_request)

    if not mcq_questions:
        raise HTTPException(status_code=500, detail="MCQ generation failed.")

    exam_id = f"exam_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    total_marks = exam_request.total_questions * exam_request.marks_per_question

    questions = []
    for q in mcq_questions:
        options = [MCQOption(**opt) for opt in q.get("options", [])]
        questions.append(MCQQuestion(
            question=q.get("question", ""),
            options=options,
            explanation=q.get("explanation", ""),
            marks=q.get("marks", exam_request.marks_per_question),
        ))

    exam_response = ExamResponse(
        exam_id=exam_id,
        exam_name=exam_request.exam_name,
        book_name=exam_request.book_name,
        date=exam_request.date,
        duration=exam_request.duration,
        total_questions=exam_request.total_questions,
        marks_per_question=exam_request.marks_per_question,
        total_marks=total_marks,
        questions=questions,
    )

    save_exam_to_file(exam_response)
    return exam_response


@prefix_router.get("/list-books/{collection_name}")
async def list_books(collection_name: str):
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = scroll_result  # Unpack tuple correctly
        books = {
            point.payload.get("book_title")
            for point in points if point.payload.get("book_title")
        }
        return {"books": list(books)}
    except Exception as e:
        logger.error(f"List books error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list books")


@prefix_router.get("/exams")
async def list_exams():
    try:
        files = os.listdir(CONFIG["EXAMS_FOLDER"])
        exams = []
        for fname in files:
            if fname.endswith(".json"):
                with open(os.path.join(CONFIG["EXAMS_FOLDER"], fname), "r") as f:
                    data = json.load(f)
                    exams.append({
                        "exam_id": data["exam_id"],
                        "exam_name": data["exam_name"],
                        "date": data["date"],
                        "total_questions": data["total_questions"],
                    })
        return {"exams": exams}
    except Exception as e:
        logger.error(f"List exams error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list exams")


@prefix_router.get("/exam/{exam_id}")
async def get_exam(exam_id: str):
    try:
        for fname in os.listdir(CONFIG["EXAMS_FOLDER"]):
            if fname.startswith(exam_id):
                with open(os.path.join(CONFIG["EXAMS_FOLDER"], fname), "r") as f:
                    return json.load(f)
        raise HTTPException(status_code=404, detail="Exam not found")
    except Exception as e:
        logger.error(f"Get exam error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get exam")

# ----------- App Start -----------

app = FastAPI(title="MCQ Exam Generator API")
app.include_router(prefix_router, prefix="/ai")
