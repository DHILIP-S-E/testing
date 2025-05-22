from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import logging
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import os
import re
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

# Initialize embedding model and Qdrant client
try:
    embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(url=CONFIG["QDRANT_URL"], api_key=CONFIG["API_KEY"])
    logger.info("Successfully initialized embedder and Qdrant client")
except Exception as e:
    logger.error(f"Failed to initialize embedder or Qdrant client: {e}")
    embedder = None
    qdrant_client = None

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

# ----------- Helpers -----------

def query_qdrant_for_book(book_name: str, collection_name: str, limit: int = 20):
    if not qdrant_client:
        logger.error("Qdrant client not initialized")
        return []

    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            logger.error(f"Collection {collection_name} does not exist")
            return []

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        logger.info(f"Querying collection: {collection_name}, vectors: {collection_info.vectors_count}")

        # Encode query
        query_embedding = list(embedder.embed([book_name]))[0].tolist()

        # First try with exact match filter
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

        # If no results, try without filter
        if not search_result:
            logger.info(f"No exact match for {book_name}, searching without filter")
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
            )

        results = [
            {
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("book_title", ""),
                "similarity": hit.score,
            }
            for hit in search_result
        ]

        logger.info(f"Found {len(results)} chunks for {book_name}")
        return results
    except Exception as e:
        logger.error(f"Qdrant query error: {e}")
        return []

def extract_json_from_text(text):
    """Extract JSON content from text that might contain other characters."""
    try:
        # Try to find a JSON array
        json_pattern = r'\[\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(?:\s*,\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*\s*\]'
        json_matches = re.findall(json_pattern, text, re.DOTALL)

        if json_matches:
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Try with basic start/end detection
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # If no valid JSON array was found, check if we have a direct JSON object
        if text.strip().startswith("{") and text.strip().endswith("}"):
            try:
                return [json.loads(text)]
            except json.JSONDecodeError:
                pass

        # Last resort: try to parse the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # If all parsing attempts fail, do manual extraction
        # For the specific format seen in the error logs
        questions = []
        question_blocks = re.findall(r'question[\'"]?\s*:\s*[\'"]([^\'"]*).*?answers[\'"]?\s*:\s*\[(.*?)\].*?correct_answer[\'"]?\s*:\s*[\'"]([^\'"]*)',
                                   text, re.DOTALL)

        for q_text, answers_text, correct_answer in question_blocks:
            # Parse answers
            answers = re.findall(r'[\'"]([^\'"]*)[\'"]', answers_text)

            # Create a proper question object
            question = {
                "question": q_text,
                "options": [],
                "explanation": f"The correct answer is: {correct_answer}",
                "marks": 5  # Default value
            }

            # Create options
            for i, answer in enumerate(answers):
                option_letter = chr(65 + i)  # A, B, C, D...
                question["options"].append({
                    "option": option_letter,
                    "text": answer,
                    "correct": (answer == correct_answer)
                })

            questions.append(question)

        if questions:
            logger.info(f"Manually extracted {len(questions)} questions")
            return questions

        logger.error("Failed to extract valid JSON")
        return None

    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return None

def convert_mistral_format_to_mcq(questions_data):
    """Convert various question formats to our standard MCQ format."""
    standardized_questions = []

    for q in questions_data:
        try:
            # Case 1: Already in our expected format
            if "question" in q and "options" in q and isinstance(q["options"], list):
                if all(isinstance(opt, dict) and "option" in opt and "text" in opt and "correct" in opt for opt in q["options"]):
                    standardized_questions.append(q)
                    continue

            # Case 2: Format with question, answers array, and correct_answer
            if "question" in q and ("answers" in q or "answer_choices" in q) and "correct_answer" in q:
                answers = q.get("answers") or q.get("answer_choices")
                correct_answer = q.get("correct_answer")

                options = []
                for i, answer in enumerate(answers):
                    option_letter = chr(65 + i)  # A, B, C, D...
                    options.append({
                        "option": option_letter,
                        "text": answer,
                        "correct": (answer == correct_answer)
                    })

                explanation = q.get("explanation", f"The correct answer is: {correct_answer}")
                marks = q.get("marks", 5)  # Default value

                standardized_questions.append({
                    "question": q["question"],
                    "options": options,
                    "explanation": explanation,
                    "marks": marks
                })
                continue

            # Case 3: Format with question, options object with lettered keys, and correct option letter
            if "question" in q and "options" in q and isinstance(q["options"], dict) and "correct" in q:
                options = []
                for opt_letter, opt_text in q["options"].items():
                    if opt_letter in ["A", "B", "C", "D"]:
                        options.append({
                            "option": opt_letter,
                            "text": opt_text,
                            "correct": (opt_letter == q["correct"])
                        })

                explanation = q.get("explanation", f"The correct answer is: {q['options'].get(q['correct'], '')}")
                marks = q.get("marks", 5)  # Default value

                standardized_questions.append({
                    "question": q["question"],
                    "options": options,
                    "explanation": explanation,
                    "marks": marks
                })
                continue

            logger.warning(f"Unrecognized question format: {q}")

        except Exception as e:
            logger.error(f"Error converting question format: {e}")

    return standardized_questions

def generate_mcqs_with_mistral(context: str, exam_request: ExamRequest, is_common_ai: bool = False):
    try:
        # Limit context length to avoid token overflow
        max_context_length = 12000
        if len(context) > max_context_length:
            logger.info(f"Truncating context from {len(context)} to {max_context_length} characters")
            context = context[:max_context_length]

        # Different prompts based on AI option
        if is_common_ai:
            prompt = f"""
IMPORTANT: You MUST generate EXACTLY {exam_request.total_questions} UNIQUE multiple-choice questions based on the following input.
I need EXACTLY {exam_request.total_questions} questions - no more, no less.

Each question MUST be different from the others - avoid repetition in both questions and answers.

Each question should include:
1. The question text
2. A list of possible answers (at least 2 options, preferably 4)
3. The correct answer

The questions should be worth {exam_request.marks_per_question} marks each.

Input to base questions on:
{context}

Format each question as a JSON object with "question", "answers" (as an array), and "correct_answer" fields.
Return EXACTLY {exam_request.total_questions} questions in a JSON array.
"""
        else:
            prompt = f"""
IMPORTANT: You MUST generate EXACTLY {exam_request.total_questions} UNIQUE multiple-choice questions about the following text.
I need EXACTLY {exam_request.total_questions} questions - no more, no less.

Each question MUST be different from the others - avoid repetition in both questions and answers.

Each question should include:
1. The question text
2. A list of possible answers (at least 2 options, preferably 4)
3. The correct answer

The questions should be worth {exam_request.marks_per_question} marks each.

Text to base questions on:
{context}

Format each question as a JSON object with "question", "answers" (as an array), and "correct_answer" fields.
Return EXACTLY {exam_request.total_questions} questions in a JSON array.
"""
        logger.info(f"Sending prompt to Mistral (length: {len(prompt)})")

        payload = {
            "model": "mistral",
            "prompt": prompt,
            "max_tokens": 12000,  # Increased to ensure we have enough space for all questions
            "temperature": 0.5,   # Increased slightly to encourage more diverse outputs
            "stream": False,
        }

        resp = requests.post(CONFIG["OLLAMA_API_URL"], json=payload, timeout=120)
        resp.raise_for_status()

        response_data = resp.json()
        response_text = response_data.get("response", "")

        logger.info(f"Received response from Mistral (length: {len(response_text)})")

        # Extract JSON from the response
        questions_data = extract_json_from_text(response_text)

        if not questions_data:
            logger.error("Failed to extract questions from Mistral response")
            # Try to create MCQs directly from the raw text as last resort
            questions_data = create_mcqs_from_raw_text(response_text, exam_request.marks_per_question)
            if not questions_data:
                return []

        # Convert any format variations to our standard MCQ format
        standardized_questions = convert_mistral_format_to_mcq(questions_data)

        # Final validation and corrections
        validated_questions = []
        seen_questions = set()  # Track unique questions to avoid duplicates

        for q in standardized_questions:
            # Ensure we have questions and options
            if "question" not in q or "options" not in q or not q["options"]:
                continue

            # Check for duplicate questions
            question_text = q["question"].strip().lower()
            if question_text in seen_questions:
                logger.info(f"Skipping duplicate question: {question_text[:50]}...")
                continue

            seen_questions.add(question_text)

            # Ensure explanation exists
            if "explanation" not in q or not q["explanation"]:
                q["explanation"] = "See the correct answer marked above."

            # Ensure marks exists
            if "marks" not in q:
                q["marks"] = exam_request.marks_per_question

            # Ensure at least one option is marked correct
            correct_count = sum(1 for opt in q["options"] if opt.get("correct", False))
            if correct_count == 0 and q["options"]:
                q["options"][0]["correct"] = True

            # If more than one correct, keep only the first
            if correct_count > 1:
                found_first = False
                for opt in q["options"]:
                    if opt.get("correct", False):
                        if found_first:
                            opt["correct"] = False
                        found_first = True

            validated_questions.append(q)

        logger.info(f"Successfully processed {len(validated_questions)} unique questions")
        return validated_questions
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Mistral API failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Mistral generation error: {e}")
        return []

def create_mcqs_from_raw_text(text: str, marks_per_question: int):
    """Create MCQs from raw text when JSON parsing fails."""
    questions = []

    # Look for numbered or Q: style questions
    question_blocks = re.split(r'\n\s*(?:\d+[\)\.]\s*|\bQ(?:uestion)?\s*(?:\d+)?[\:\)\.]\s*)', text)

    for block in question_blocks:
        if not block.strip():
            continue

        # Try to identify question text and answers
        parts = re.split(r'\n\s*(?:[Oo]ptions|[Aa]nswers|[Cc]hoices)[\:\)\.]\s*', block, 1)
        if len(parts) < 2:
            continue

        question_text = parts[0].strip()
        options_text = parts[1]

        # Extract options (A, B, C, D style)
        options = []
        options_matches = re.findall(r'\b([A-D])[\.\)]\s*([^\n]+?)(?=\n\s*[A-D][\.\)]|\n\s*(?:Answer|Correct)|\Z)', options_text)

        for opt_letter, opt_text in options_matches:
            options.append({
                "option": opt_letter,
                "text": opt_text.strip(),
                "correct": False  # Default to false, will set correct one below
            })

        # Find correct answer
        correct_match = re.search(r'\b(?:Answer|Correct)[^\n]*?:\s*([A-D])', options_text)
        if correct_match and options:
            correct_letter = correct_match.group(1)
            for opt in options:
                if opt["option"] == correct_letter:
                    opt["correct"] = True
                    break
        elif options:  # If can't find correct answer, default to A
            options[0]["correct"] = True

        # If we have a valid question with options
        if question_text and options:
            questions.append({
                "question": question_text,
                "options": options,
                "explanation": "See the correct answer marked above.",
                "marks": marks_per_question
            })

    return questions

def save_exam_to_file(exam_data: ExamResponse):
    try:
        filename = f"{exam_data.exam_id}_{exam_data.exam_name.replace(' ', '_')}.json"
        filepath = os.path.join(CONFIG["EXAMS_FOLDER"], filename)
        with open(filepath, "w", encoding="utf-8") as f:
            # Use model_dump() instead of deprecated dict() method
            json.dump(exam_data.model_dump(), f, indent=2)
        logger.info(f"Saved exam to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save exam: {e}")
        return None

def create_default_mcq_options(answers, correct_answer):
    """Create standard MCQ options from simple answers list and correct answer."""
    options = []
    for i, answer in enumerate(answers):
        option_letter = chr(65 + i)  # A, B, C, D...
        options.append({
            "option": option_letter,
            "text": answer,
            "correct": (answer == correct_answer)
        })
    return options

# ----------- Routes -----------

@prefix_router.post("/generate-questions", response_model=ExamResponse)
async def generate_exam(exam_request: ExamRequest):
    logger.info(f"Generating exam: {exam_request.exam_name} for book: {exam_request.book_name}, AI option: {exam_request.ai_option}")

    # Check if using Common AI or Medi AI
    is_common_ai = exam_request.ai_option.lower() == "common ai"

    if not is_common_ai and (not embedder or not qdrant_client):
        raise HTTPException(
            status_code=500,
            detail="Embedder or Qdrant client not initialized. Check server logs."
        )

    # For Common AI, we use the book_name directly as input context
    if is_common_ai:
        logger.info(f"Using Common AI to generate questions based on input: {exam_request.book_name}")
        context = exam_request.book_name

    # For Medi AI, we fetch content from Qdrant
    else:
        logger.info(f"Using Medi AI to generate questions from Qdrant books")

        # Check if collection_name is provided
        if not exam_request.collection_name:
            # If collection_name is not provided, use book_name directly as context
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

            # Create context from chunks
            context = "\n\n".join([chunk["text"] for chunk in book_chunks])
            logger.info(f"Created context with {len(context)} characters from {len(book_chunks)} chunks")

    # Generate MCQs with appropriate flag for Common AI
    mcq_questions = generate_mcqs_with_mistral(context, exam_request, is_common_ai)

    # If we didn't get enough unique questions, try again with a higher request count
    # Try up to 3 times to get the requested number of questions
    max_retries = 3
    retry_count = 0

    while len(mcq_questions) < exam_request.total_questions and retry_count < max_retries:
        retry_count += 1
        logger.info(f"Got only {len(mcq_questions)} unique questions, trying again (attempt {retry_count}/{max_retries})")

        # Create a copy of the exam request with increased question count
        retry_request = ExamRequest(
            exam_name=exam_request.exam_name,
            book_name=exam_request.book_name,
            date=exam_request.date,
            duration=exam_request.duration,
            total_questions=exam_request.total_questions * (retry_count + 1),  # Increase multiplier with each retry
            marks_per_question=exam_request.marks_per_question,
            ai_option=exam_request.ai_option,
            collection_name=exam_request.collection_name  # This will be None if not provided
        )

        # Try again with the higher count
        new_questions = generate_mcqs_with_mistral(context, retry_request, is_common_ai)

        # Add new questions to our existing set, avoiding duplicates
        existing_question_texts = {q.get("question", "").strip().lower() for q in mcq_questions}
        for q in new_questions:
            question_text = q.get("question", "").strip().lower()
            if question_text and question_text not in existing_question_texts:
                mcq_questions.append(q)
                existing_question_texts.add(question_text)

        # If we now have enough questions, break out of the loop
        if len(mcq_questions) >= exam_request.total_questions:
            break

    if not mcq_questions:
        raise HTTPException(
            status_code=500,
            detail="MCQ generation failed. Check server logs for details."
        )

    exam_id = f"exam_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Process questions
    questions = []
    for q in mcq_questions:
        try:
            # Handle the case where we don't have options in standard format
            if "options" not in q:
                if "answers" in q and "correct_answer" in q:
                    # Convert from answers + correct_answer format to options format
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

            # Skip questions with no options
            if not options:
                logger.warning(f"Question has no valid options: {q}")
                continue

            # Add missing options if we have less than 4
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

    # If no valid questions were created
    if not questions:
        raise HTTPException(
            status_code=500,
            detail="Failed to create valid questions from generated content."
        )

    # Limit to requested number of questions
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


@prefix_router.get("/list-books/{collection_name}")
async def list_books(collection_name: str):
    if not qdrant_client:
        raise HTTPException(
            status_code=500,
            detail="Qdrant client not initialized. Check server logs."
        )

    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist"
            )

        # Get book titles
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = scroll_result  # Unpack tuple

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


@prefix_router.get("/collections")
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


@prefix_router.get("/exams")
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


@prefix_router.get("/exam/{exam_id}")
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

        # If we get here, no matching exam was found
        raise HTTPException(status_code=404, detail=f"Exam with ID '{exam_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get exam error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get exam: {str(e)}")

# ----------- Health check endpoint -----------

@prefix_router.get("/health")
async def health_check():
    health = {
        "status": "ok" if embedder and qdrant_client else "error",
        "embedder": "initialized" if embedder else "error",
        "qdrant": "connected" if qdrant_client else "error",
        "timestamp": datetime.now().isoformat()
    }
    return health
