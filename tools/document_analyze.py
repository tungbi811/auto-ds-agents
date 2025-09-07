import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter

def analyze_pdf_from_path(file_path):
    """Analyze PDF from file path"""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    
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
    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def handle_pdf_analysis(message, make_ai_request_func, get_prompt_config_func, client, file_path=None):
    """Handle PDF analysis requests"""
    try:
        if not file_path:
            return {"response": "Error: No PDF file provided for analysis.", "search_used": False}
        
        # Analyze the PDF
        analysis_results = analyze_pdf_from_path(file_path)
        
        # Join text chunks if it's a list
        if isinstance(analysis_results, list):
            analysis_text = " ".join(analysis_results)
        else:
            analysis_text = str(analysis_results)

        # Get prompt configuration and format it
        prompt_config = get_prompt_config_func("pdf_analysis")
        prompt = prompt_config["template"].format(analysis_results=analysis_text, message=message)

        response = make_ai_request_func(prompt, "pdf_analysis", client)
        return {"response": response, "search_used": False}
        
    except Exception as e:
        return {"response": f"PDF analysis failed: {str(e)}", "search_used": False}