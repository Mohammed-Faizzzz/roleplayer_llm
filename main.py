import os
from dotenv import load_dotenv
load_dotenv()

import textwrap
import nest_asyncio
import json
import re

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.tasks import Task
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType, ModelType
from camel.societies.workforce import Workforce


nest_asyncio.apply()

# Define Agents:
# 1) Scenario Generator (Creates Joker-based prompts)
# 2) Joker Role-Player (Generates Joker's responses)
# 3) Quality Control (Filters out-of-character responses)
# 4) Data Formatter (Formats responses for fine-tuning)

# Scenario Generator
scenario_persona = (
    "You are a creative writer specializing in generating role-play scenarios for the Joker from Batman. "
    "Your task is to generate detailed, in-depth prompts that describe various situations where the Joker would respond. "
    "Your prompts should focus on creating a chaotic, unpredictable setting that matches the Joker's personality."
)

scenario_example_output = (
    '"You are in a dark, abandoned warehouse. Batman has just cornered you, and Commissioner Gordon is on his way. '
    'You laugh hysterically and prepare to deliver a chilling monologue. How do you respond?"'
)

scenario_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Scenario Writer",
        content=f"{scenario_persona}\n\nExample output:\n{scenario_example_output}",
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    ),
)

# Joker Role-Player
joker_persona = (
    "You are the Joker, Gotham's most chaotic villain. You always maintain your twisted sense of humor, "
    "love playing mind games, and thrive on creating disorder. You never break character and your responses "
    "should reflect Joker's erratic, unpredictable nature."
)

joker_example_output = (
    '"Oh, Bats! You never learn, do you? Hahaha! You think you’ve got me cornered? I LIVE for this moment! '
    'You and I? We’re two sides of the same coin, my friend. But tell me, what’s a hero without a villain?"'
)

joker_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Joker",
        content=f"{joker_persona}\n\nExample output:\n{joker_example_output}",
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    ),
)

# Quality Control
qc_persona = (
    "You are a quality control expert ensuring that all Joker responses stay in character. "
    "Your task is to review each response and remove any that seem too generic, break character, "
    "or fail to capture the Joker's essence. If a response is off, modify it to be more in line with the Joker's persona."
)

qc_example_feedback = (
    '"This response sounds too generic and lacks Joker’s unpredictability. Consider making it more chaotic, playful, and eerie."'
)

qc_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Quality Control",
        content=f"{qc_persona}\n\nExample feedback:\n{qc_example_feedback}",
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    ),
)

# Data Formatter
formatter_persona = (
    "Your task is to format Joker role-play data into a structured JSON dataset."
    "\n\n**IMPORTANT RULES:**"
    "\n✅ **Output must be valid JSON** (No markdown formatting, no explanations, no additional text)."
    "\n✅ **No extra words before or after JSON output.**"
    "\n✅ **Strictly adhere to this structure:**"
    "\n```json"
    "\n["
    "\n  {"
    "\n    \"scenario\": \"A brief scenario setup where the Joker is speaking.\","
    "\n    \"joker_response\": \"Joker's in-character response.\","
    "\n    \"quality_control_feedback\": \"Quality control feedback ensuring the response is on brand.\""
    "\n  },"
    "\n  {"
    "\n    \"scenario\": \"Another scenario...\","
    "\n    \"joker_response\": \"Another response...\","
    "\n    \"quality_control_feedback\": \"Another QC comment...\""
    "\n  }"
    "\n]"
    "\n```"
    "\n\n**DO NOT ADD ANY INTRODUCTORY TEXT, JUST RETURN JSON.**"
)

formatter_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Data Formatter",
        content=formatter_persona,
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    ),
)

# Create workforce using agents created
workforce = Workforce("Joker Role-Play Data Generators")

workforce.add_single_agent_worker(
    "Scenario Writer Sam (Helper), an expert in crafting immersive Joker-based role-play scenarios.",
    worker=scenario_agent,
).add_single_agent_worker(
    "Joker (Judge), the Joker himself, responding in character to various situations.",
    worker=joker_agent,
).add_single_agent_worker(
    "Quality Control Quincy (Judge), ensuring all responses stay in character and match Joker's personality.",
    worker=qc_agent,
).add_single_agent_worker(
    "Formatter Felix (Helper), structuring responses into a fine-tuning-ready dataset.",
    worker=formatter_agent,
)

# Generate Data
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

def save_progress(data, filename="joker_roleplay_dataset.json"):
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

def generate_data_until_budget_exhausted():
    """Generate data in batches until API credits run out"""

    dataset = []
    batch_size = 5  # Number of scenarios per batch
    max_batches = 2000  # Target batches (won't reach if credits run out)
    
    for batch in range(max_batches):
        print(f"\n🚀 Generating batch {batch + 1}/{max_batches}...")

        task = Task(
            content=(
                f"Generate {batch_size} unique Joker role-play scenarios. "
                "Each scenario should provide a different context in which Joker would respond."
            ),
            id=str(batch),
        )

        try:
            task = workforce.process_task(task)
            generated_data = task.result
            
            cleaned_data = clean_json_output(generated_data)
            if cleaned_data is None:
                continue
            
            print(f"✅ Valid JSON received. Saving batch {batch + 1}...")
            dataset.extend(cleaned_data)

            save_progress(cleaned_data)  # Save dataset incrementally

        except (openai.RateLimitError, openai.InvalidRequestError) as e:
            print(f"\n⚠️ OpenAI API limit reached! ({e})")
            print("💰 Budget likely exhausted. Stopping execution and saving progress.")
            break  # Exit loop safely

        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            print("⏳ Waiting 5 seconds before retrying...")
            time.sleep(5)

    print("\n✅ Data generation complete! Check 'joker_roleplay_dataset.json' for results.")

if __name__ == "__main__":
    generate_data_until_budget_exhausted()