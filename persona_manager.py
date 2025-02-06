import random
import difflib
from camel.tasks import Task
from agent_creation import workforce


# This is a base list of personas. Ideally, when generating 1000+ data points, we will need
# 100+ personas of varying personalities so that the model will be able to generalise the
# personality traits for new characters not included in the dataset based on their interpreted personality
available_personas = [
    {"name": "Joker", "description": "Chaotic, unpredictable, theatrical, dark humor."},
    {"name": "Sherlock Holmes", "description": "Logical, deductive, observant, precise."},
    {"name": "Yoda", "description": "Wise, speaks in inverted syntax, cryptic, mentor-like."},
    {"name": "Tony Stark", "description": "Sarcastic, witty, confident, tech-savvy."},
    {"name": "Darth Vader", "description": "Deep voice, commanding presence, ruthless."},
    {"name": "Gandalf", "description": "Mysterious, wise, speaks in old English style."},
    {"name": "Walter White", "description": "Calculated, scientific, ego-driven, manipulative."},
    {"name": "Hannibal Lecter", "description": "Polite, intellectual, eerie, sophisticated."},
    {"name": "Jack Sparrow", "description": "Drunk, charismatic, deceptive, pirate-like speech."},
    {"name": "Deadpool", "description": "Meta, irreverent, comedic, fourth-wall breaking."}
]
persona_queue = available_personas.copy()

def find_closest_persona(persona_name):
    """Finds the closest matching persona based on the given persona name."""
    persona_names = [p["name"] for p in available_personas]
    closest_match = difflib.get_close_matches(persona_name, persona_names, n=1, cutoff=0.6)
    if closest_match:
        return next((p for p in available_personas if p["name"] == closest_match[0]), None)
    return None

def generate_persona_description(persona_name):
    """If a persona is not found, it queries the Large LLM to generate a description for it to roleplay.
    Ideally, this should not be required during inference, because we would have a good enough dataset.
    Alternatively, we should make the LLM to instead query the user for the description"""
    print(f"⚠️ Persona '{persona_name}' not found. Generating description...")
    
    persona_prompt = (
        f"Describe {persona_name} in 2 sentences. Include their personality, speech style, and moral stance."
    )

    # Call an LLM to generate a new persona description dynamically
    new_description = workforce.process_task(
        Task(content=persona_prompt, id="generate_persona")
    ).result

    return new_description.strip()

    __all__ = ["available_personas", "persona_queue", "find_closest_persona", "generate_persona_description"]
