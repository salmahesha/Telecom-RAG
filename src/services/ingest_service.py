from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore.database import VectorDatabase
from src.logging.logger import logger

class IngestionService:
    def __init__(self):
        self.repo = VectorDatabase()
    async def process_uploaded_file(self , file:UploadFile , chunk_size = 500 , chunk_overlap = 100):
        logger.info(f"Upload file Content{file.filename}");
        content = await file.read()
        text_content = content.decode("utf-8").replace("\r\n" , "\n")
        
        logger.info("===== Splitting content ====")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =chunk_size,
            chunk_overlap =chunk_overlap,
            length_function =len,
            separators=["\n\n" , "\r\n\r\n" , "\n" , " " , ""]
        )
        docs = text_splitter.create_documents(
            texts=[text_content],
            metadatas=[{"source": file.filename}]
        )
        logger.info(f"Generated {len(docs)} text chunks from '{file.filename}'.")

        self.repo.create_from_documents(docs)
        return len(docs)
    