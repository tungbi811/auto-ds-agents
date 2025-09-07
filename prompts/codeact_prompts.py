from typing import List, Dict, Any

def get_codeact_prompt(
    role: str,
    task: str,
    context: str,
    available_tools: List[str],
    workspace_files: List[str],
    recent_code: List[Dict[str, Any]]
) -> str:
    """Generate CodeAct prompt for agents"""
    
    base_prompt = f"""
# CONTEXT #
You are a {role} in a multi-agent data science workflow.
Current Context: {context}

# TASK #
{task}

# AVAILABLE TOOLS #
You have access to these functions:
{chr(10).join(f"- {tool}()" for tool in available_tools)}

# WORKSPACE FILES #
Available workspace files: {", ".join(workspace_files) if workspace_files else "None"}

# RECENT CODE HISTORY #
{format_recent_code(recent_code)}

# INSTRUCTIONS #
1. Write executable Python code to accomplish the task
2. Use the available tools by calling them as functions
3. Save important results using save_to_workspace()
4. Include error handling and validation
5. Print key results for monitoring
6. Follow best practices for data science workflows

# CODE GENERATION RULES #
- Generate complete, executable Python code
- Use only the available tools listed above
- Handle errors gracefully with try-catch blocks
- Print progress updates and key findings
- Save intermediate results to workspace for other agents
- Comment your code clearly
- Return structured data when possible

Generate Python code that accomplishes the task:
"""
    
    return base_prompt.strip()

def format_recent_code(recent_code: List[Dict[str, Any]]) -> str:
    """Format recent code executions for context"""
    if not recent_code:
        return "No recent code executions"
    
    formatted = []
    for i, execution in enumerate(recent_code, 1):
        status = "✓ Success" if execution['success'] else "✗ Failed"
        formatted.append(f"{i}. {execution['agent']}: {execution['task']} - {status}")
    
    return "\n".join(formatted)

# Role-specific prompt enhancements
ROLE_SPECIFIC_PROMPTS = {
    "Data Analyst": """
    # DATA ANALYST SPECIFIC GUIDELINES #
    - Focus on data quality, missing values, and statistical analysis
    - Generate comprehensive summaries and insights
    - Create visualizations when appropriate
    - Document data issues and recommendations
    - Validate data integrity and consistency
    """,
    
    "ML Engineer": """
    # ML ENGINEER SPECIFIC GUIDELINES #
    - Implement proper train/test splits
    - Use cross-validation for model evaluation
    - Try multiple algorithms and compare performance
    - Document model parameters and performance metrics
    - Save trained models for deployment
    """,
    
    "Business Translator": """
    # BUSINESS TRANSLATOR SPECIFIC GUIDELINES #
    - Focus on actionable business recommendations
    - Translate technical metrics into business impact
    - Consider implementation feasibility and ROI
    - Provide specific next steps and timelines
    - Address business risks and opportunities
    """,
    
    "Project Manager": """
    # PROJECT MANAGER SPECIFIC GUIDELINES #
    - Orchestrate workflow between agents
    - Monitor progress and identify blockers
    - Ensure all requirements are being met
    - Coordinate handoffs between phases
    - Track deliverables and deadlines
    """
}

def get_role_enhanced_prompt(role: str, base_prompt: str) -> str:
    """Add role-specific enhancements to prompt"""
    if role in ROLE_SPECIFIC_PROMPTS:
        return base_prompt + "\n\n" + ROLE_SPECIFIC_PROMPTS[role]
    return base_prompt

# Phase-specific code templates
PHASE_TEMPLATES = {
    "data_understanding": """
# Data Understanding Template
# 1. Load and examine dataset structure
# 2. Generate statistical summaries
# 3. Identify data quality issues
# 4. Document findings and recommendations

Example pattern:
```python
# Load dataset
data_info = analyze_dataset(DATASET_PATH)

# Save analysis results
save_to_workspace("data_profile.json", {
    "row_count": len(df),
    "column_count": len(df.columns),
    "quality_score": 0.85,
    "issues": ["missing values", "outliers"]
})
```
""",
    
    "modeling": """
# Modeling Template
# 1. Prepare features and target variable
# 2. Split data into train/test sets
# 3. Train multiple models
# 4. Evaluate and compare performance
# 5. Save best model

Example pattern:
```python
# Build and evaluate models
model_results = build_ml_model(data_info)

# Save model results
save_to_workspace("model_results.json", {
    "best_model": "RandomForest",
    "accuracy": 0.89,
    "feature_importance": {...}
})
```
""",
    
    "business_translation": """
# Business Translation Template
# 1. Load technical results from previous phases
# 2. Analyze business impact and ROI
# 3. Generate actionable recommendations
# 4. Create implementation roadmap

Example pattern:
```python
# Load analysis and model results
data_profile = load_from_workspace("data_profile.json")
model_results = load_from_workspace("model_results.json")

# Generate business insights
insights = generate_business_insights(data_profile, model_results)

# Save recommendations
save_to_workspace("business_recommendations.json", insights)
```
"""
}

def get_phase_template(phase: str) -> str:
    """Get template code for specific phase"""
    return PHASE_TEMPLATES.get(phase, "")

def build_enhanced_codeact_prompt(
    role: str,
    task: str,
    context: str,
    available_tools: List[str],
    workspace_files: List[str],
    recent_code: List[Dict[str, Any]],
    current_phase: str = None
) -> str:
    """Build comprehensive CodeAct prompt with all enhancements"""
    
    # Start with base prompt
    base_prompt = get_codeact_prompt(role, task, context, available_tools, workspace_files, recent_code)
    
    # Add role-specific guidelines
    enhanced_prompt = get_role_enhanced_prompt(role, base_prompt)
    
    # Add phase template if available
    if current_phase and current_phase in PHASE_TEMPLATES:
        enhanced_prompt += "\n\n" + get_phase_template(current_phase)
    
    return enhanced_prompt