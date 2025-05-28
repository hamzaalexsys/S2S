import PyPDF2
import io
from typing import Optional
import streamlit as st

class PDFProcessor:
    def extract_text(self, pdf_file) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None 