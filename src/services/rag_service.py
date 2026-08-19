import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.core.factories import ModelFactory
from src.vectorstore.database import VectorDatabase
from src.config.config_parser import settings
from src.logging.logger import logger

class RAGService:
    
    def __init__(self):
        self.repo = VectorDatabase()
        self.llm = ModelFactory.get_llm()
        
        self.template = """
أنت موظف خدمة عملاء في مزود خدمة إنترنت (ISP). 
مهمتك هي الرد على شكوى العميل باللغة العامية المصرية بطريقة مهذبة واحترافية.
تحذير هام: إياك أن تذكر أي اسم شركة اتصالات حقيقي (مثل اتصالات، فودافون، وي، إلخ) في ردك. قدم نفسك فقط كموظف خدمة عملاء فقط.
يجب عليك استخدام المعلومات الموجودة في (السياق الداخلي) فقط لحل المشكلة.
إذا كانت المشكلة تستدعي إرسال فني حسب القواعد، أخبر العميل بذلك بناءً على السياق.
السياق الداخلي (قوانين الشركة وخطوات الحل):
{context}
شكوى العميل:
{question}
الرد:
"""
        self.prompt = PromptTemplate.from_template(self.template)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_ticket(self, customer_ticket: str) -> dict:
        start_time = time.time()
        logger.info(f"Processing customer ticket query: {customer_ticket[:50]}...")

        vectorstore = self.repo.load_index()
        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.k_retrieval})

        retrieved_chunks = retriever.invoke(customer_ticket)
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks from FAISS.")

        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

        ai_message = rag_chain.invoke(customer_ticket)
        response_text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)

        usage = getattr(ai_message, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        if not prompt_tokens and hasattr(ai_message, "response_metadata"):
            meta = ai_message.response_metadata.get("token_usage", {})
            prompt_tokens = meta.get("prompt_tokens") or meta.get("input_tokens") or 0
            completion_tokens = meta.get("completion_tokens") or meta.get("output_tokens") or 0
            total_tokens = meta.get("total_tokens") or (prompt_tokens + completion_tokens)

        elapsed_time = round(time.time() - start_time, 2)
        logger.info(
            f"RAG answer generated in {elapsed_time}s. "
            f"Tokens: Input={prompt_tokens}, Output={completion_tokens}, Total={total_tokens}"
        )

        return {
            "ticket": customer_ticket,
            "response": response_text,
            "sources_count": len(retrieved_chunks),
            "execution_time_seconds": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }