import re
from autogen import GroupChatManager

PHASES = ["BusinessAnalyst", "DataExplorer", "DataEngineer", "ModelBuilder", "Evaluator", "BusinessTranslator"]

ALLOWED_LOOPS = {
    ("DataExplorer", "BusinessAnalyst"),
    ("ModelBuilder", "DataEngineer"),
    ("Evaluator", "BusinessAnalyst"),
}

LOOP_PATTERN = re.compile(r"<<REQUEST_LOOP:(?P<to>[A-Za-z0-9_]+)\|(?P<reason>.+?)>>")
STOP_TOKEN = "<END_OF_WORKFLOW>"

class Manager(GroupChatManager):
    def __init__(self, group_chat, llm_config):
        super().__init__(
            groupchat=group_chat,
            llm_config=llm_config,
            system_message=f"""
                You are the Manager Agent in a CRISP-DM multi-agent workflow.

                Enforce the standard order:
                1) BusinessAnalyst → 2) DataExplorer → 3) DataEngineer → 4) ModelBuilder → 5) Evaluator → 6) BusinessTranslationAgent

                Feedback loops (allowed backtracks):
                - DataExplorer → BusinessAnalyst  (data doesn't support goals / need to refine objectives)
                - ModelBuilder → DataEngineer     (data prep/features/transforms insufficient)
                - Evaluator → BusinessAnalyst     (results don't meet business success criteria)

                Code execution protocol:
                - If an agent's JSON output contains a top-level field "code" (string) is not null, route to CodeExecutor.

                How agents request a loop:
                - An agent that needs to loop back MUST end its message with:
                  <<REQUEST_LOOP:TargetAgent|short reason>>
                  Example: <<REQUEST_LOOP:BusinessAnalyst|data lacks churn labels>>

                Rules:
                - If loop request is valid (allowed pair), route to the requested TargetAgent next.
                - If request is invalid, ignore it, reply asking the same agent to correct it, and stay on the same phase.
                - If an agent's JSON output is invalid/incomplete, ask the SAME agent to fix it before moving on.
                - Pass prior agents' JSON outputs verbatim when they are referenced.
                - Coordinate only; do NOT produce domain content yourself.

                Auto-stop:
                - When BusinessTranslator posts the final business summary, it must end with the token: {STOP_TOKEN}
                - As soon as a message contains that token, stop the conversation.
            """
        )

    # Helper used by a speaker_selection_method to choose who speaks next.
    def route_next(self, agents, last_speaker, last_message: str):
        # Stop if final token present
        if STOP_TOKEN in (last_message):
            return None  # Autogen will terminate if selection returns None

        # Loop request?
        m = LOOP_PATTERN.search(last_message)
        if m:
            target = m.group("to")
            if (last_speaker, target) in ALLOWED_LOOPS:
                # Find and return the requested agent if present
                for a in agents:
                    if a.name == target:
                        return a
                # If not found, fall through to sequential routing

        # Default sequential routing
        try:
            idx = PHASES.index(last_speaker)
            # Move forward, cap at last phase
            next_name = PHASES[min(idx + 1, len(PHASES) - 1)]
        except ValueError:
            # If last speaker not in PHASES (e.g., user), start from first phase
            next_name = PHASES[0]

        for a in agents:
            if a.name == next_name:
                return a
        
        print("No next speaker selected")
        # Fallback: no match → end
        return None