from autogen import OpenAIWrapper

def is_ready_for_train(groupchat):
    client = OpenAIWrapper(
        api_type='openai', 
        model='gpt-4o-mini',
    )
    
    messages = (
        [
            {
                "role": "system",
                "content": """
                                Based on the dataset exploration, and the data processing, please determine whether the data is ready for model training.
                                Please give a short summary of what we know about the dataset and what we have done so far.

                                Please follow this format:
                                Summary: <Your summary>
                                Decision: <choose from "Ready for training" or "Need more processing">
                            """,
            }
        ] + groupchat.messages
    )

    response = client.create(messages=messages)
    response_str = client.extract_text_or_completion_object(response)[0]

    print('-'*50)
    print(response_str)
    print('-'*50)

    if "ready for training" in response_str.lower():
        return True
    return False

def count_train_trials(groupchat):
    messages = groupchat.messages

    count = 0
    for i, message in enumerate(messages):
        if message['name'] == 'Model_Trainer':
            count += 1
        elif (
            message['name'] == 'Code_Executor'
            and "exitcode: 1" in message["content"]
            and messages[i - 1]["name"] == "Model_Trainer"
        ):
            count -= 1

    return count

def custom_speaker_selection_func(last_speaker, groupchat):
    messages = groupchat.messages
    
    if last_speaker.name == 'user':
        return groupchat.agent_by_name('Data_Explorer')

    # these states contains two steps, we will always call code_executor after the first step
    elif last_speaker.name in ['Data_Explorer', 'Data_Processor', 'Model_Trainer']:
        return groupchat.agent_by_name('Code_Executor')

    elif last_speaker.name == 'Code_Executor':
        last_second_speaker_name = groupchat.messages[-2]["name"]

        # if we get an error, we repeat the current state
        if "exitcode: 1" in messages[-1]["content"]:
            return groupchat.agent_by_name(last_second_speaker_name)

        # explore state
        elif last_second_speaker_name == "Data_Explorer":
            return groupchat.agent_by_name('Data_Processor')

        # process state
        elif last_second_speaker_name == "Data_Processor":
            if is_ready_for_train(groupchat=groupchat):
                return groupchat.agent_by_name('Model_Trainer')
            return groupchat.agent_by_name('Data_Explorer')

        elif last_second_speaker_name == "Model_Trainer":
            if count_train_trials(groupchat) < 2:
                return groupchat.agent_by_name('Model_Trainer')
            return groupchat.agent_by_name('Code_Summarizer')

    # summarize state
    elif last_speaker.name == "Code_Summarizer":
        return None  # end the conversation
    
def save_agent_code(chat_result):

    if "```python" in chat_result.chat_history[-1]["content"]:
        content = chat_result.chat_history[-1]["content"]
        content = content.split("```python")[1].split("```")[0].strip()
        with open("./agent_code/house_price_prediction.py", "w") as f:
            f.write(content)