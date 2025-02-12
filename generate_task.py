import random
from camel.tasks import Task
from persona_manager import available_personas, persona_queue, find_closest_persona, generate_persona_description

TRAITS = {
    "tone": ["formal", "casual", "sarcastic", "mysterious", "intellectual", "aggressive"],
    "speech_pace": ["fast", "slow", "deliberate", "erratic"],
    "word_choice": ["sophisticated", "plainspoken", "poetic", "slang-heavy", "technical"],
    "logic_style": ["emotional", "rational", "manipulative", "cynical", "idealistic"]
}

SCENARIO_ELEMENTS = {
    "setting": [
        "a cyberpunk city", "a medieval castle", "a courtroom", "a deep-sea base",
        "a futuristic utopia", "a deserted spaceship", "an interdimensional gateway"
    ],
    "conflict": [
        "falsely accused of a crime", "forced to make a moral decision", 
        "dealing with a betrayal", "caught in a time loop", "leading a rebellion",
        "discovering they are not who they think they are", "debating an AI about free will"
    ],
    "twist": [
        "they have lost their memory", "they are secretly being watched", 
        "someone they trust is actually an enemy", "they must work with their worst enemy",
        "they must convince a skeptic they are telling the truth",
        "they must complete a heist without being detected", "they are forced into an absurd situation"
    ],
    "interaction_type": [
        "casual banter", "philosophical debate", "interrogation",
        "unexpected comedy", "strategic negotiation", "power struggle"
    ]
}

def generate_persona():
    """Dynamically creates a new persona based on random traits."""
    name = f"Generated Persona {random.randint(1000, 9999)}"
    description = (
        f"This persona speaks in a **{random.choice(TRAITS['tone'])}** tone, "
        f"with a **{random.choice(TRAITS['speech_pace'])}** speech pace, "
        f"using **{random.choice(TRAITS['word_choice'])}** vocabulary, "
        f"and prefers **{random.choice(TRAITS['logic_style'])}** reasoning."
    )
    return {"name": name, "description": description}

def generate_scenario():
    """Creates a procedurally varied scenario dynamically"""
    setting = random.choice(SCENARIO_ELEMENTS["setting"])
    conflict = random.choice(SCENARIO_ELEMENTS["conflict"])
    twist = random.choice(SCENARIO_ELEMENTS["twist"])
    interaction = random.choice(SCENARIO_ELEMENTS["interaction_type"])

    return (
        f"The character finds themselves in {setting}. They must handle a situation where they are {conflict}. "
        f"However, {twist}. The tone of this conversation should be a **{interaction}**."
    )


def generate_task(batch, batch_size, dynamic_ratio=0.3):
    """
    Creates tasks while mixing existing characters with dynamically generated ones.
    `dynamic_ratio` controls how many characters are newly generated per batch.
    """
    global persona_queue

   
    # num_dynamic = int(batch_size * dynamic_ratio) # should be 1 for batch_size = 3
    # num_predefined = batch_size - num_dynamic # should be 2 for batch_size = 3
    num_dynamic, num_predefined = 1, 2 # hardcoded for batch size = 3 for now

    # Select some predefined personas
    selected_personas = []
    if len(persona_queue) < batch_size:
        persona_queue = available_personas.copy()  # Refresh the queue and shuffle
        random.shuffle(persona_queue)
    for _ in range(num_dynamic):
        selected_personas.append(generate_persona())
    for _ in range(num_predefined):
        persona = persona_queue.pop()
        selected_personas.append(persona)

    task_content = "Generate unique role-play dialogues for the following characters.\n"
    task_content += "Ensure each dialogue is distinct from previous ones by:\n"
    task_content += "- Placing the character in **a unique, never-before-used setting**.\n"
    task_content += "- Introducing an **unexpected challenge** for the persona.\n"
    task_content += "- Making sure no line repeats previous interactions.\n\n"

    for persona in selected_personas:
        unique_scenario = generate_scenario()  # Generate a fresh scenario

        task_content += (
            f"- **Persona: {persona['name']}**\n"
            f"  **Description:** {persona['description']}\n"
            f"  **Scenario:** {unique_scenario}\n"
            f"  **Dialogue:** Generate a 12-turn role-play conversation featuring {persona['name']} responding naturally.\n"
            f"  **Speech Style Requirement:** {persona['name']} must respond in their distinct tone and speaking style (e.g., poetic, sarcastic, highly logical, broken syntax, overly dramatic)."
        )

    return Task(
        content=task_content,
        id=str(batch),
    )
