# embedding_creator.py
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

def create_embeddings(texts):
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  return embeddings.embed_documents(texts)