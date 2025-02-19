# import json
# import os

# def clean_json_output(json_text):
#     """Cleans JSON output by removing Markdown formatting and ensuring valid JSON."""
#     json_text = json_text.strip()
#     if json_text.startswith("```json"):
#         json_text = json_text[7:-3].strip()  # Remove Markdown JSON formatting
#     try:
#         return json.loads(json_text)
#     except json.JSONDecodeError:
#         return None

# def save_progress(data, filename="roleplay_dataset.json"):
#     """Saves the generated JSON data to a file."""
#     if not data:
#         return

#     # Append data to existing JSON file
#     if os.path.exists(filename) and os.path.getsize(filename) > 0:
#         with open(filename, "r", encoding="utf-8") as f:
#             existing_data = json.load(f)
#     else:
#         existing_data = []

#     existing_data.extend(data)

#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(existing_data, f, indent=4)

#     print(f"Progress saved! Dataset now contains {len(existing_data)} samples.")

import json
import os
import re

from sentence_transformers import SentenceTransformer, util

# Code from this file is still buggy, hence requiring manual copy pasting of JSON output to data file

def clean_json_output(json_text):
    """Strips markdown formatting and ensures JSON is properly parsed with detailed debugging."""
    
    # 🚨 Check if empty response
    if not json_text or json_text.strip() == "":
        print("❌ Received empty response! Skipping batch...")
        return None

    json_text = json_text.strip()

    # 🚀 STEP 2: Remove Markdown JSON formatting (if present)
    # json_text = re.sub(r"^```json\s*", "", json_text, flags=re.MULTILINE)
    # json_text = re.sub(r"\s*```$", "", json_text, flags=re.MULTILINE)

    try:
        # 🚀 STEP 3: First attempt at JSON parsing
        parsed_json = json.loads(json_text)
        
        # 🚨 Check if it’s a string (means it's **double-encoded** JSON)
        if isinstance(parsed_json, str):
            parsed_json = json.loads(parsed_json)

        # 🚨 Ensure it's a list (our expected format)
        if not isinstance(parsed_json, list):
            return None
        return parsed_json

    except json.JSONDecodeError as e:
        return None

def save_progress(data, filename="roleplay_dataset.json"):
    """Ensures only valid JSON is saved, removing Markdown formatting if needed"""
    try:
        
        if isinstance(data, str):
            data = clean_json_output(data)
            if data is None:
                return
        
        # Append data to existing JSON file
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

        print(f"Progress saved! Dataset now contains {len(existing_data)} samples.")

    except json.JSONDecodeError as e:
        print(f"Error: JSON file is still corrupted. Details: {e}")

model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast

def is_duplicate(new_text, dataset_texts, threshold=0.85):
    new_embedding = model.encode(new_text, convert_to_tensor=True)
    dataset_embeddings = model.encode(dataset_texts, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(new_embedding, dataset_embeddings)

    if max(similarity_scores[0]) > threshold:
        return True  # Reject: Too similar
    return False  # Accept: Unique


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


import json

def load_dataset(input_filename):
    """
    Load dataset from a JSON file.
    
    Args:
        input_filename (str): The path to the JSON file to load.
        
    Returns:
        list: The loaded dataset.
    """
    with open(input_filename, 'r') as file:
        return json.load(file)

def save_dataset(output_filename, data):
    """
    Save the converted dataset to a JSON file.
    
    Args:
        output_filename (str): The path to the output JSON file.
        data (list): The data to save.
    """
    with open(output_filename, 'w') as file:
        json.dump(data, file, indent=4)

# Assuming `convert_dataset_to_chat_format` is already defined
# Read the dataset from 'dataset.json'
input_data = load_dataset('dataset.json')

# Convert the dataset to chat format
converted_data = convert_dataset_to_chat_format(input_data)

# Save the converted dataset to 'output.json'
save_dataset('output.json', converted_data)

print("Dataset successfully converted and saved to 'output.json'.")

