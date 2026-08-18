# import os
# absolute_path = os.path.abspath(__file__)

# print(f"absolute_path:{absolute_path} " )

from fastapi import FastAPI
from src.config.config_parser import settings
from src.logging.logger import logger
from src.core.factories import ModelFactory

async def running(app:FastAPI):
    logger.info("====================================\n ")
    logger.info(f"=== Starting {settings.app_name} v{settings.app_version} ===")
    logger.info("Warming up ML Models (Singleton Embeddings & LLM)...")
    ModelFactory.get_embeddings()
    ModelFactory.get_llm()
    logger.info("====================================\n ")
    yield  
    logger.info("=== Shutting down Telecom RAG Microservice ===")


app = FastAPI(
    title = settings.app_name,
    version= settings.app_version,
    description= "Telecom RAG With FAST API",
    lifespan=running
)

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000)