from .business_analyst import BusinessAnalyst
from .business_translator import BusinessTranslator
from .coder import Coder
from autogen import UserProxyAgent
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import ContextVariables, RevertToUserTarget, AgentTarget

class GroupChat:
    def __init__(self):
        context_variables = ContextVariables(data={
            "current_agent": "",
            "objective": "",
            "problem_type": "",
            "stakeholders_expectations": [],
            "research_questions": [],
        })
    
        coder = Coder()
        business_analyst = BusinessAnalyst()
        business_translator = BusinessTranslator()
        user = UserProxyAgent(
            name="User",
            code_execution_config=False
        )

        business_translator.handoffs.set_after_work(RevertToUserTarget())
        user.handoffs.set_after_work(AgentTarget(business_analyst))

        self.pattern = DefaultPattern(
            initial_agent=business_analyst,
            agents=[business_analyst, coder, business_translator],
            user_agent=user,
            context_variables=context_variables,
        )

    def run(self, dataset_paths, user_requirements):
        message = f"""
            Data path: {dataset_paths}
            Requirements: {user_requirements}
        """

        response = run_group_chat(
            pattern=self.pattern,
            messages=message,
            max_rounds=200
        )

        return response.events