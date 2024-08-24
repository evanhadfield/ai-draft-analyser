# pdf_processor.py
import requests
import io
from PyPDF2 import PdfReader

def extract_text_from_pdf_url(url):
  response = requests.get(url)
  if response.status_code == 200:
      pdf_file = io.BytesIO(response.content)
      pdf_reader = PdfReader(pdf_file)
      num_pages = len(pdf_reader.pages)
      text = ""
      for i, page in enumerate(pdf_reader.pages, 1):
          text += page.extract_text()
      return text, num_pages
  else:
      return None, 0

def extract_text_from_pdf_file(file):
  pdf_reader = PdfReader(file)
  num_pages = len(pdf_reader.pages)
  text = ""
  for i, page in enumerate(pdf_reader.pages, 1):
      text += page.extract_text()
  return text, num_pages