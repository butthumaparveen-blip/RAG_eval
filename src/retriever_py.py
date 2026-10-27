from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import List
import os
DB_folder="chroma_db"
#load pdf it can be multiple
def load_pdf():
   data=PyPDFLoader(r"C:\Users\butth\Documents\RAG_practice\RAG\data\Pakistan_road_safety_profile_2025 (1).pdf")
   doc=data.load()
   return doc

def load_vectorstore():
   embeddings=OllamaEmbeddings(model="nomic-embed-text")
   if os.path.exists(DB_folder):
      return Chroma(embedding_function=embeddings,persist_directory=DB_folder)
   else:
      doc=load_pdf()
      splitter=RecursiveCharacterTextSplitter(
         chunk_size=800,
         chunk_overlap=100,
      )
      chunks=splitter.split_documents(doc)
      vector_store=Chroma.from_documents(
         embedding=embeddings,
         documents=chunks,
         persist_directory=DB_folder
      )
      return vector_store
def retriever_system():
   return load_vectorstore().as_retriever(
      search_type="mmr",
      search_kwargs={
         "fetch_k":6,
         "k":4
      }
   )
   
if __name__=="__main__":
  retriever=retriever_system()
  res=retriever.invoke("How many road-crash deaths did WHO estimate in Pakistan in 2021?")
  print(res)
  