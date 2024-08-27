# Civis AI Draft Analyzer

This repository contains the Civis AI-powered Draft Analyzer, designed to provide various scores after analyzing PDF documents. The objective is to deliver accurate scores, along with explanations for the scores and recommendations for improvement.

## Platform Architecture

1. **Upload or Input PDF**:
   - Upload the PDF file or enter the URL of the PDF.
2. **Document Processing**:
   - Split the document into chunks using a text splitter.
   - Upload the chunks into the Pinecone Vector Database.
3. **Embedding and Scoring**:
   - Use OpenAI embeddings for storing data in the vector database.
   - Run the RAG (Retrieval-Augmented Generation) scorer to generate scores.
   - Optionally, enhance the score further using prompt engineering and few-shot examples from Langchain.

## Tech Stack

- **Languages/Frameworks**: Python, Streamlit
- **Database**: Pinecone Vector Database

## System Requirements

- **Python**: Version 3.0 and above

## Deployment

- **Platform**: Streamlit

## Credentials

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `PINECONE_API_KEY`

## Running Locally

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>

   ```

2. Set Up Environment:

Create a `.env` file and provide the necessary credentials as mentioned above.

Create a virtual environment and install all dependencies from `requirements.txt`

Run the Application:
`streamlit run app.py`
