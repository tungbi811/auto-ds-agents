import re
from autogen import GroupChatManager

PHASES = [
    "BusinessAnalyst",
    "DataExplorer",
    "DataEngineer",
    "ModelBuilder",
    "Evaluator",
    "BusinessTranslationAgent",
]

# Feedback loops allowed in CRISP-DM
ALLOWED_LOOPS = {
    ("DataExplorer", "BusinessAnalyst"),
    ("ModelBuilder", "DataEngineer"),
    ("Evaluator", "BusinessAnalyst"),
}

LOOP_PATTERN = re.compile(r"<<REQUEST_LOOP:(?P<to>[A-Za-z0-9_]+)\|(?P<reason>.+?)>>")
STOP_TOKEN = "<END_OF_WORKFLOW>"

class Manager(GroupChatManager):
    def __init__(self, group_chat, llm_config, stop_token: str = STOP_TOKEN):
        self.stop_token = stop_token
        self._pending_executor_return_to = None
        super().__init__(
            groupchat=group_chat,
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message=f"""
                You are the Manager Agent.
                Coordinate the CRISP-DM workflow and route agents correctly.

                Order:
                1) BusinessAnalyst → 2) DataExplorer → 3) DataEngineer → 4) ModelBuilder → 5) Evaluator → 6) BusinessTranslationAgent

                Feedback loops allowed:
                - DataExplorer → BusinessAnalyst
                - ModelBuilder → DataEngineer
                - Evaluator → BusinessAnalyst

                Special rules:
                - If a message ends with <RUN_THIS> or contains a fenced code block,
                route to CodeExecutor, then return back to the requester.
                - If a message ends with <<REQUEST_LOOP:TargetAgent|reason>>,
                and the loop is valid, route there.
                - If the BusinessTranslationAgent ends its message with {stop_token}, stop the workflow.

                You do not generate content yourself; only coordinate agents.
            """
        )

    def route_next(self, agents, last_speaker, last_message: str):
        # 1. Stop if final token present
        if self.stop_token in (last_message or ""):
            return None

        # 2. If CodeExecutor just spoke, return to requester
        if last_speaker == "CodeExecutor" and self._pending_executor_return_to:
            target_name = self._pending_executor_return_to
            self._pending_executor_return_to = None
            return next((a for a in agents if a.name == target_name), None)

        # 3. Explicit code execution request
        if "<RUN_THIS>" in (last_message or "") or "```" in (last_message or ""):
            for a in agents:
                if a.name == "CodeExecutor":
                    self._pending_executor_return_to = last_speaker
                    return a

        # 4. Feedback loop request
        # m = LOOP_PATTERN.search(last_message or "")
        # if m:
        #     target = m.group("to")
        #     if (last_speaker, target) in ALLOWED_LOOPS:
        #         return next((a for a in agents if a.name == target), None)
        #     # invalid loop → retry same agent
        #     return next((a for a in agents if a.name == last_speaker), None)

        # 5. Default sequential order
        try:
            idx = PHASES.index(last_speaker)
            next_name = PHASES[min(idx + 1, len(PHASES) - 1)]
        except ValueError:
            next_name = PHASES[0]

        return next((a for a in agents if a.name == next_name), None)
