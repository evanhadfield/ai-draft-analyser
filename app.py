# app.py
import streamlit as st
from pdf_processor import extract_text_from_pdf_url, extract_text_from_pdf_file
from text_processor import clean_text, split_text
from embedding_creator import create_embeddings
from pinecone_uploader import upload_to_pinecone
from rag_scorer import run_rag_scorer


st.title("Civis-AI Powered Draft Analyser")

upload_method = st.radio("Choose upload method:", ("Upload PDF", "Enter PDF URL"))

if upload_method == "Upload PDF":
  uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
  if uploaded_file is not None:
      text, num_pages = extract_text_from_pdf_file(uploaded_file)
      st.success(f"PDF uploaded successfully. Number of pages: {num_pages}")
else:
  url = st.text_input("Enter PDF URL (ending with .pdf)")
  if url and url.endswith('.pdf'):
      text, num_pages = extract_text_from_pdf_url(url)
      if text:
          st.success(f"PDF downloaded successfully. Number of pages: {num_pages}")
      else:
          st.error("Failed to download the PDF. Please check the URL.")
  elif url:
      st.warning("Please enter a valid PDF URL ending with .pdf")

if 'text' in locals() and text:
  st.write(f"Extracted text length: {len(text)} characters")
  
  cleaned_text = clean_text(text)
  st.write(f"Cleaned text length: {len(cleaned_text)} characters")
  
  if st.button("Process and Upload"):
      with st.spinner("Processing and uploading..."):
          chunks = split_text(cleaned_text)
          embeddings = create_embeddings(chunks)
          upload_to_pinecone(chunks, embeddings)
      st.success("Text processed and uploaded successfully!")
  if st.button("Run Scorer"):
    with st.spinner("Running Scorer..."):
        result = run_rag_scorer()
    
    st.success("Scoring complete!")
    
    # Split the result into separate components for each category
    lines = result.split("\n")
    
    for line in lines:
        if "Score:" in line and "Reason:" in line:
            try:
                score_part, reason_part = line.split("Reason:")
                
                # Display the score on a separate line
                st.write(score_part.strip())
                
                # Display the reason on a new line
                st.write(f"Reason: {reason_part.strip()}")
            except ValueError:
                st.error("Error processing line: " + line)
        else:
            st.write(line.strip())
