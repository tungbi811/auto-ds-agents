from crewai import Agent, Task, LLM
import os
from dotenv import load_dotenv
from crewai.project import CrewBase, agent, task
from streamlit import json

from crew.crew_tools import get_tools_for_agent, set_session_id

load_dotenv()

@CrewBase
class MultiAgent:
    """Multi-agent workflow with chatbot requirements integration"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(model="openai/gpt-3.5-turbo")

    def __init__(self, session_id=None):
            # Use provided session ID
            if session_id:
                set_session_id(session_id)
                print(f"Initialized workflow with session: {session_id}")
            else:
                print("No session ID provided to MultiAgent")

    @agent
    def business_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["business_analyst"],
            tools=get_tools_for_agent("business_analyst"),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
            handle_parsing_errors=True
        )

    @agent
    def project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["project_manager"],
            tools=get_tools_for_agent("project_manager"),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=10,
            handle_parsing_errors=True
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            tools=get_tools_for_agent("data_analyst"),
            llm=self.llm,
            verbose=True,
            allow_code_execution=True,
            max_iter=10,
            handle_parsing_errors=True
        )

    @agent
    def ml_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_engineer"],
            tools=get_tools_for_agent("ml_engineer"),
            llm=self.llm,
            verbose=True,
            allow_code_execution=True,
            max_iter=10,
            handle_parsing_errors=True
        )

    @agent
    def business_translator(self) -> Agent:
        return Agent(
            config=self.agents_config["business_translator"],
            tools=get_tools_for_agent("business_translator"),
            llm=self.llm,
            verbose=True,
            max_iter=10,
            handle_parsing_errors=True
        )

    @task
    def understand_task(self) -> Task:
        return Task(
            config=self.tasks_config["business_context_analysis"],
            agent=self.business_analyst(),
            tools=get_tools_for_agent("business_analyst"),
            output_file="/workspace/business_issue.md"
        )

    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["planning"],
            agent=self.project_manager(),
            tools=get_tools_for_agent("project_manager"),
            context=[self.understand_task()],
            output_file="/workspace/todo.md"
        )

    @task
    def data_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis"],
            agent=self.data_analyst(),
            tools=get_tools_for_agent("data_analyst"),
            context=[self.understand_task(), self.plan_task()],
            output_file="/workspace/data_analysis_report.md",
            max_execution_time=600,  # 10 minutes
            max_retry=3
        )     
    
    @task
    def model_task(self) -> Task:
        return Task(
            config=self.tasks_config["modeling"],
            agent=self.ml_engineer(),
            tools=get_tools_for_agent("ml_engineer"),
            context=[self.data_task()],
            output_file="/workspace/model_report.md",
            max_execution_time=600,  # 10 minutes
            max_retry=3
        )

    @task
    def translate_task(self) -> Task:
        return Task(
            config=self.tasks_config["business_translation"],
            agent=self.business_translator(),
            tools=get_tools_for_agent("business_translator"),
            context=[self.model_task()],
            output_file="/workspace/business_recommendations.md"
        )

