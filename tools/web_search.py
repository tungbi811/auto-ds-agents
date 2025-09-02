import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

def web_search(query, num_results=5):
    search_results = DDGS().text(query, max_results=num_results)
    return search_results

def handle_web_search(message, make_ai_request_func, get_prompt_config_func, client):
    """Handle web search requests"""
    try:
        search_results = web_search(message)

        # Get prompt configuration and format it
        prompt_config = get_prompt_config_func("web_search")
        prompt = prompt_config["template"].format(search_results=search_results, message=message)

        response = make_ai_request_func(prompt, "web_search")
        return {"response": response, "search_used": True}
    except Exception as e:
        return {"response": f"Search failed: {str(e)}. Here's a regular AI response instead.", "search_used": False}