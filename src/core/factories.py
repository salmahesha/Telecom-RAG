"""
    #! for instantiating ML models
"""

import os
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.config_parser import settings
from src.logging.logger import logger

class ModelFactory:
    
    @staticmethod
    @lru_cache(maxsize=1) # to use same object
    def get_embeddings():
        if settings.embedding_provider.lower() == "huggingface":
            logger.info(f"Embdding model provider:: {settings.embedding_provider}")
            return HuggingFaceEmbeddings(
                model_name = settings.embedding_model_name,
                model_kwargs = {'device' : 'cpu'},
                encode_kwargs ={'normalize_embeddings' : True}
            )
        else:
            raise ValueError(f"Not support this embedding provide: {settings.embedding_provider}")
        
        
    @staticmethod
    @lru_cache(maxsize=1) # to use same object
    def get_llm():
        if settings.llm_provider.lower() == "gemini":
            logger.info(f"LLM provider:: {settings.llm_provider}")
            return ChatGoogleGenerativeAI(
                model = settings.llm_model_name,
                google_api_key=settings.google_api_key,
                temperature = settings.temperature # 
            )
        else:
            raise ValueError(f"Not support this llm provider: {settings.llm_provider}")
        
        