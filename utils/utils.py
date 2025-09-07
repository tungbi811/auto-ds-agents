from typing import Dict, Any

def make_codeact_request(prompt: str, client, model: str = "gpt-3.5-turbo", temperature: float = 0.1) -> str:
    """Make AI request optimized for code generation"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

def get_agent_model_config(agent_role: str) -> Dict[str, Any]:
    """Get model configuration for specific agent roles"""
    configs = {
        "Data Analyst": {"model": "gpt-3.5-turbo", "temperature": 0.3},
        "ML Engineer": {"model": "gpt-3.5-turbo", "temperature": 0.1},
        "Business Translator": {"model": "gpt-3.5-turbo", "temperature": 0.5},
        "Project Manager": {"model": "gpt-3.5-turbo", "temperature": 0.2}
    }
    return configs.get(agent_role, {"model": "gpt-3.5-turbo", "temperature": 0.3})

def safe_json_parse(content: str, default: Any = None) -> Any:
    """Safely parse JSON content with fallback"""
    try:
        import json
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return default