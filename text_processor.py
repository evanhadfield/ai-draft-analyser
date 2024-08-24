# text_processor.py
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
  text = re.sub(r'--- Page \d+ ---', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  hindi_range = r'\u0900-\u097F'
  pattern = fr'[^a-zA-Z0-9\s{hindi_range}]'
  text = re.sub(pattern, '', text)
  return text

def split_text(text):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
  )
  return text_splitter.split_text(text)