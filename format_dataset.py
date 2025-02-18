import json

def convert_dataset_to_chat_format(input_data):
    """
    input_data: a list of data points, each data point is structured like:
        {
            "persona": "Archmage Zephyrus",
            "description": "A wise wizard...",
            "dialogue": [
                {"speaker": "Archmage Zephyrus", "text": "First line..."},
                {"speaker": "Tempest", "text": "Second line..."},
                ...
            ]
        }
    
    Returns a list of new data points in user–assistant style. 
    Each new data point is an array of messages, e.g.:
        [
          {"role": "system",    "content": "..."},
          {"role": "assistant", "content": "..."},
          {"role": "user",      "content": "..."},
          ...
        ]
    """
    new_data = []

    for entry in input_data:
        persona = entry.get("persona", "Unknown Persona")
        description = entry.get("description", "")
        dialogue = entry.get("dialogue", [])

        # Start by constructing a conversation list
        conversation = []

        # 1) Add a system message giving context about the persona
        system_content = (
            f"You are {persona}. Description: {description}\n"
            "Stay in character and respond accordingly."
        )
        conversation.append({
            "role": "system",
            "content": system_content
        })

        # 2) Convert each speaker turn into either 'assistant' or 'user'
        for turn in dialogue:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")

            if speaker == persona:
                # This line is from the persona, treat it as the "assistant"
                role = "assistant"
            else:
                # Everyone else is the "user"
                role = "user"

            conversation.append({
                "role": role,
                "content": text
            })

        # Append the converted conversation to new_data
        new_data.append(conversation)

    return new_data

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

converted_data = convert_dataset_to_chat_format(data)

with open("chat_dataset.json", "w", encoding="utf-8") as f_out:
        json.dump(converted_data, f_out, indent=2, ensure_ascii=False)
