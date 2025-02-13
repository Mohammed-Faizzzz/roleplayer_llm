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
