from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.config.config_parser import settings
from src.logging.logger import logger
from src.core.factories import ModelFactory
from src.vectorstore.database import VectorDatabase
from src.routers import ingest_router, query_router

async def running(app:FastAPI):
    logger.info("====================================\n ")
    logger.info(f"=== Starting {settings.app_name} v{settings.app_version} ===")
    logger.info("Warming up ML Models (Singleton Embeddings & LLM)...")
    ModelFactory.get_embeddings()
    ModelFactory.get_llm()
    logger.info("====================================\n ")
    
    try:
        repo = VectorDatabase()
        repo.load_index()
        logger.info("vectors loaded in memory")
    except Exception as ex :
        logger.warning(f"vectors not loaded {str(ex)}")
    yield  
    logger.info("=== Shutting down Telecom RAG Microservice (pause) ===")


app = FastAPI(
    title = settings.app_name,
    version= settings.app_version,
    description= "Telecom RAG With FAST API",
    lifespan=running
)



app.include_router(ingest_router.router)
app.include_router(query_router.router)

@app.get('/')
def welcome():
    return {
        "status": 200,
        "message":"Server is running",
        "app_name" :settings.app_name,
        "version" :settings.app_version,
        "URL":"/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000 , reload=True)