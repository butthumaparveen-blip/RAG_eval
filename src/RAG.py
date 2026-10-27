from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.retriever_py import retriever_system
from src.generator_code import generate_answer 
class RAG_building:
    def __init__(self,):
      self.retriever=retriever_system()
    def invoke(self,query):
       documents=self.retriever.invoke(query)
       context = [doc.page_content for doc in documents]
       generated_answer=generate_answer(query,context)
       return {
          "query":query,
          "context":context,
          "generated_answer":generated_answer
       }


