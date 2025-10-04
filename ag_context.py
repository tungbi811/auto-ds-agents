from typing import Annotated
from datetime import datetime
from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ReplyResult, AgentNameTarget, ContextVariables
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize context variables
support_context = ContextVariables(data={
    "session_start": datetime.now().isoformat(),
    "query_history": [],
    "solutions_provided": [],
    "query_count": 0,
    "solution_count": 0
})

# Define tools that use context variables
def classify_and_log_query(
    query: Annotated[str, "The user query to classify"],
    context_variables: ContextVariables
) -> ReplyResult:
    """Classify a user query as technical or general and log it in the context."""

    # Printing the context variables
    print(f"{context_variables.to_dict()=}")

    # Record this query in history
    query_history = context_variables.get("query_history", [])
    query_record = {
        "timestamp": datetime.now().isoformat(),
        "query": query
    }
    query_history.append(query_record)
    context_variables["query_history"] = query_history
    context_variables["last_query"] = query
    context_variables["query_count"] = len(query_history)

    # Basic classification logic
    technical_keywords = ["error", "bug", "broken", "crash", "not working", "shutting down"]
    is_technical = any(keyword in query.lower() for keyword in technical_keywords)

    # Update context with classification
    if is_technical:
        target_agent = AgentNameTarget("tech_agent")
        context_variables["query_type"] = "technical"
        message = "This appears to be a technical issue. Routing to technical support..."
    else:
        target_agent = AgentNameTarget("general_agent")
        context_variables["query_type"] = "general"
        message = "This appears to be a general question. Routing to general support..."

    # Printing the context variables
    print(f"{context_variables.to_dict()=}")

    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables
    )

def provide_technical_solution(
    solution: Annotated[str, "Technical solution to provide"],
    context_variables: ContextVariables
) -> ReplyResult:
    """Provide a technical solution and record it in the context."""

    # Printing the context variables
    print(f"{context_variables.to_dict()=}")

    # Record the solution
    last_query = context_variables.get("last_query", "your issue")
    solutions_provided = context_variables.get("solutions_provided", [])

    solution_record = {
        "timestamp": datetime.now().isoformat(),
        "query": last_query,
        "solution": solution
    }
    solutions_provided.append(solution_record)

    # Update context
    context_variables["solutions_provided"] = solutions_provided
    context_variables["last_solution"] = solution
    context_variables["solution_count"] = len(solutions_provided)

    # Printing the context variables
    print(f"{context_variables.to_dict()=}")

    return ReplyResult(
        message=solution,
        context_variables=context_variables
    )

def get_user_context(context_variables: ContextVariables) -> str:
    """Get information about the current user and session."""
    user_name = context_variables.get("user_name", "User")
    query_count = context_variables.get("query_count", 0)
    session_start = context_variables.get("session_start", "Unknown")

    return f"""Current session information:
    - Session started: {session_start}
    - Total queries: {query_count}
    - Last query: {context_variables.get("last_query", "None")}
    - Query type: {context_variables.get("query_type", "Unknown")}"""

# Create the agents
llm_config = llm_config = LLMConfig(config_list={"api_type": "openai", "model": "gpt-5-nano","api_key":os.getenv("OPENAI_API_KEY")})

triage_agent = ConversableAgent(
    name="triage_agent",
    system_message="""You are a triage agent. For each user query,
    identify whether it is a technical issue or a general question.
    Use the classify_and_log_query tool to categorize and log queries.""",
    functions=[classify_and_log_query],
    llm_config=llm_config
)

tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs
    and hardware issues. After analyzing the problem, use the provide_technical_solution
    tool to format your response consistently and log it for future reference.""",
    functions=[provide_technical_solution],
    # Using UpdateSystemMessage to make context visible
    update_agent_state_before_reply=[
        UpdateSystemMessage(
            "You are assisting with query #{query_count}. "
            "The current issue is: {last_query} (Type: {query_type})"
        )
    ],
    llm_config=llm_config
)

general_agent = ConversableAgent(
    name="general_agent",
    system_message="""You handle general, non-technical support questions.
    Use the get_user_context tool to understand the current session before responding.""",
    # Providing a tool to access context
    functions=[get_user_context],
    llm_config=llm_config
)

# User agent
user = ConversableAgent(name="user", human_input_mode="ALWAYS")

# Set up the conversation pattern with context variables
pattern = AutoPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    context_variables=support_context,  # Pass our initialized context
    group_manager_args={"llm_config": llm_config}
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="My laptop keeps shutting down randomly. Can you help?",
    max_rounds=10
)