'''
# ? Keep configuration separate from code
# ? Read from Config.yml
'''
import os
from dotenv import load_dotenv
import yaml
load_dotenv()

class Config:
    def __init__(self , configuration_path:str=None):
        if configuration_path is None:
            absolute_path = os.path.abspath(__file__)
            # Telecom Rag
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(absolute_path)))
            configuration_path = os.path.join(base_dir , 'config' , 'config.yml')
            
        with open(configuration_path , 'r' , encoding="utf-8") as file:
            self._config = yaml.safe_load(file)

    
    
    # as attribute --> setting.app_name
    @property
    def app_name(self) -> str:
        return self._config["app"]["name"]

    @property
    def app_version(self) -> str:
        return self._config["app"]["version"]

    @property
    def vector_index_path(self) -> str:
        return self._config["data"]["vector_index_path"]

    @property
    def chunk_size(self) -> int:
        return int(self._config["data"]["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return int(self._config["data"]["chunk_overlap"])

    @property
    def batch_size(self) -> int:
        return int(self._config["data"].get("batch_size", 50))

    @property
    def embedding_provider(self) -> str:
        return self._config["models"]["embedding_provider"]

    @property
    def embedding_model_name(self) -> str:
        return self._config["models"]["embedding_model_name"]

    @property
    def llm_provider(self) -> str:
        return self._config["models"]["llm_provider"]

    @property
    def llm_model_name(self) -> str:
        return self._config["models"]["llm_model_name"]

    @property
    def temperature(self) -> float:
        return float(self._config["models"]["temperature"])

    @property
    def k_retrieval(self) -> int:
        return int(self._config["models"]["k_retrieval"])

    @property
    def google_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set in .env file!")
        return key

settings = Config()