
from contextlib import asynccontextmanager
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.rag_service import RAGService
from app.ingestion.indexer import DocumentIndexer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    app.state.rag_service = RAGService()
    app.state.document_indexer = DocumentIndexer()

    yield


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description=(
        "Ask questions about company documents "
        "and receive answers with source references."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The employee's question",
    )


@app.get("/")
def home():
    return {
        "message": "Enterprise RAG Knowledge Assistant API is running."
    }


@app.get("/health")
def health(request: Request):
    service = getattr(request.app.state, "rag_service", None)

    return {
        "status": "healthy" if service else "starting",
        "service": "Enterprise RAG Knowledge Assistant",
    }



@app.post("/ask")
def ask_question(payload: AskRequest, request: Request):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        service = request.app.state.rag_service
        return service.ask(question)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=500,
            detail="Question answering failed. Check server logs.",
        )


@app.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    original_filename = Path(
        (file.filename or "").replace("\\", "/")
    ).name

    if (
        not original_filename
        or original_filename in {".", ".."}
        or "\x00" in original_filename
    ):
        raise HTTPException(
            status_code=400,
            detail="A valid filename is required.",
        )

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Only PDF, TXT, and DOCX files are supported.",
        )

    document_id = uuid.uuid4().hex
    stored_path = UPLOAD_DIR / f"{document_id}{extension}"
    total_bytes = 0

    try:
        # Stream the file to disk and enforce the size limit.
        with stored_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 10 MB limit.",
                    )

                destination.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        indexer = request.app.state.document_indexer

        result = await run_in_threadpool(
            indexer.index_document,
            stored_path,
            original_filename,
            document_id,
        )

        return {
    **result,
    "document_id": document_id,
}

    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise

    except ValueError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    
    except Exception:
        logger.exception(
            "Document upload/indexing failed for %s",
            original_filename,
        )
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Upload or indexing failed. Check server logs.",
        )


    finally:
        await file.close()
