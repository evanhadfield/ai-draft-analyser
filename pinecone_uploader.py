# pinecone_uploader.py
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def delete_all_vectors():
    try:
        index.delete(delete_all=True, namespace="")
        print("All vectors deleted from the index.")
    except Exception as e:
        print(f"Warning: Could not delete vectors: {e}")

def upload_to_pinecone(texts, text_embeddings):
    # Split the vectors into smaller batches to avoid exceeding the 2MB limit
    batch_size = 100  # Adjust this value as needed
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = text_embeddings[i:i+batch_size]
        
        vectors = [
            (f"doc_{j}", embedding, {"text": text})
            for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings), start=i)
        ]
        
        index.upsert(vectors=vectors, namespace="")
        print(f"Uploaded batch of {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'")
    
    print(f"Finished uploading all vectors to Pinecone index '{PINECONE_INDEX_NAME}'")