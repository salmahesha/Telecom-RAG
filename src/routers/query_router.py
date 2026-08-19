from fastapi import APIRouter, HTTPException, status
from src.models.schemas import QueryRequest, QueryResponse
from src.services.rag_service import RAGService
from src.logging.logger import logger

router = APIRouter(prefix="/api/v1", tags=["RAG Queries"])
rag_service = RAGService()

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def process_query(request: QueryRequest):
    try:
        if not request.ticket.strip(): # remove spaces and check if empty
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticket query cannot be empty.")
            
        result = rag_service.answer_ticket(request.ticket)
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"FAISS Index missing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector database index not found. Please trigger /api/v1/ingest first."
        )
    except Exception as e:
        logger.error(f"RAG Execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}"
        )