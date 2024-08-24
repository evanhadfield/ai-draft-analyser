# pinecone_uploader.py
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def delete_all_vectors():
  index.delete(delete_all=True)
  print("All vectors deleted from the index.")

def upload_to_pinecone(texts, text_embeddings):
  delete_all_vectors()
  vectors = [
      (f"doc_{i}", embedding, {"text": text})
      for i, (text, embedding) in enumerate(zip(texts, text_embeddings))
  ]
  index.upsert(vectors=vectors)
  print(f"Uploaded {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'")