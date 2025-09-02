import os
import sys
import PyPDF2
from flask import request
from langchain.text_splitter import CharacterTextSplitter

def analyze_uploaded_pdf():
    """Handle PDF file upload and analysis"""
    if 'file' not in request.files:
        return ["Error: No file uploaded"]
    
    file = request.files['file']
    if file.filename == '':
        return ["Error: No file selected"]
    
    if not file.filename.lower().endswith('.pdf'):
        return ["Error: Please upload a PDF file"]
    
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        if not text or len(text.strip()) < 10:
            return ["Error: Could not extract readable text from PDF"]
        
        # Split into chunks for processing
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_text(text)
    
    except PyPDF2.errors.PdfReadError:
        return ["Error: Could not read PDF file"]
    except Exception as e:
        return [f"Error: {str(e)}"]

def analyze_pdf_from_path(file_path):
    """Alternative function for analyzing PDF from file path"""
    if not file_path.endswith('.pdf') or '..' in file_path:
        raise ValueError("Invalid file path")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        if not text or len(text.strip()) < 10:
            return ["Error: Could not extract readable text from PDF"]
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_text(text)
    
    except FileNotFoundError:
        return ["Error: File not found"]
    except PyPDF2.errors.PdfReadError:
        return ["Error: Could not read PDF file"]

def handle_pdf_analysis(message, make_ai_request_func, get_prompt_config_func, client, file_path=None, use_uploaded_file=False):
    """Handle PDF analysis requests"""
    try:
        if use_uploaded_file:
            # For Flask web interface with file upload
            analysis_results = analyze_uploaded_pdf()
        elif file_path:
            # For CLI with file path
            analysis_results = analyze_pdf_from_path(file_path)
        else:
            return {"response": "Error: No PDF file provided for analysis.", "search_used": False}

        # Join text chunks if it's a list
        if isinstance(analysis_results, list):
            analysis_text = " ".join(analysis_results)
        else:
            analysis_text = str(analysis_results)

        # Get prompt configuration and format it
        prompt_config = get_prompt_config_func("pdf_analysis")
        prompt = prompt_config["template"].format(analysis_results=analysis_text, message=message)

        response = make_ai_request_func(prompt, "pdf_analysis")
        return {"response": response, "search_used": False}
    except Exception as e:
        return {"response": f"PDF analysis failed: {str(e)}", "search_used": False}

