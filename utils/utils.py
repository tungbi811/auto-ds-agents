import yaml

# Load prompts once at module level
try:
    with open("prompts/prompt.yaml", "r") as f:
        PROMPTS = yaml.safe_load(f)
except FileNotFoundError:
    PROMPTS = {}

def get_prompt_config(prompt_name):
    """Get prompt template and config from YAML"""
    return PROMPTS.get(prompt_name, {})

def make_ai_request(prompt, prompt_type="general", client=None):
    """Centralized function to make AI requests with prompt configs"""
    if not client:
        raise ValueError("OpenAI client is required")
        
    prompt_config = get_prompt_config(prompt_type)
    
    model = prompt_config.get("model", "gpt-3.5-turbo")
    temperature = prompt_config.get("temperature", 0.7)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content