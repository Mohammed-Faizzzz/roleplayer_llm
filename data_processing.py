import json
import os
import re

# Code from this file is still buggy, hence requiring manual copy pasting of JSON output to data file

def clean_json_output(json_text):
    """Strips markdown formatting (```json ... ```) from LLM output and ensures valid JSON.
       This is in place just in case the data formatter fails to format correctly."""
    json_text = json_text.strip()
    
    # Remove ```json and ``` from response if present
    json_text = re.sub(r"^```json\s*", "", json_text, flags=re.MULTILINE)
    json_text = re.sub(r"\s*```$", "", json_text, flags=re.MULTILINE)

    # This mostly still gave me invalid JSON - I had to use the JSON output in the CLI instead and manually curate dataset
    try:
        parsed_json = json.loads(json_text)
        return parsed_json
    except json.JSONDecodeError:
        print("❌ Warning: LLM did not return valid JSON. Skipping this batch.")
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

        print(f"✅ Progress saved! Dataset now contains {len(existing_data)} samples.")

    except json.JSONDecodeError as e:
        print(f"❌ Error: JSON file is still corrupted. Details: {e}")
