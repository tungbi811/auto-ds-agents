from utils import get_prompt_config, make_ai_request
from tools import handle_web_search, handle_pdf_analysis, handle_code_execution, handle_basic_eda


def agent_workflow(message, client, file_path=None, use_uploaded_file=False):
    """Enhanced agent with better intent detection"""
    
    intents = {
        "web_search": ["search", "google", "find", "latest", "current", "news"],
        "code_execute": ["calculate", "compute", "solve", "plot", "graph", "math"],
        "pdf_analysis": ["analyze pdf", "read pdf", "pdf", "document"],
        "data_analysis": ["csv", "data analysis", "eda", "dataset"]
    }
    
    message_lower = message.lower()
    
    for intent, keywords in intents.items():
        if any(keyword in message_lower for keyword in keywords):
            if intent == "web_search":
                return handle_web_search(message, make_ai_request, get_prompt_config, client)
            elif intent == "code_execute":
                return handle_code_execution(message, make_ai_request, get_prompt_config, client)
            elif intent == "pdf_analysis":
                return handle_pdf_analysis(message, make_ai_request, get_prompt_config, client, 
                                          file_path=file_path, use_uploaded_file=use_uploaded_file)
            elif intent == "data_analysis":
                return handle_basic_eda(message, None, None, None, file_path=file_path)

    # Default to regular chat
    response = make_ai_request(message, "general", client)
    return {"response": response, "search_used": False}