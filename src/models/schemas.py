from pydantic import BaseModel , Field # like validation


class IngestResponse(BaseModel):
    message:str
    chunks_indexed:int = Field(...,ge=0)# greater than 0
    index_path:str

class QueryRequest(BaseModel):
    ticket:str = Field(...,min_length=1 , max_length=100 , description="Customer ticket problem description" , example="النت عندي فى مشكلة وعايز اعرف السبب")
    
class QueryResponse(BaseModel):
    ticket: str
    response: str
    sources_count: int
    execution_time_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int