import random
from camel.tasks import Task
from persona_manager import available_personas, persona_queue, find_closest_persona, generate_persona_description

def generate_task(batch, batch_size):
    global persona_queue

    # Shuffle personas if exhausted
    if len(persona_queue) < batch_size:
        persona_queue = available_personas.copy()
        random.shuffle(persona_queue)

    selected_personas = []
    for _ in range(batch_size):
        persona = persona_queue.pop()

        # If the persona is missing, generate one dynamically
        if persona["name"] not in [p["name"] for p in available_personas]:
            closest_match = find_closest_persona(persona["name"])
            if closest_match:
                persona["description"] = closest_match["description"]
            else:
                persona["description"] = generate_persona_description(persona["name"])

        selected_personas.append(persona)

    task_content = "Generate role-play dialogues for the following characters:\n\n"

    for persona in selected_personas:
        task_content += (
            f"- **Persona: {persona['name']}**\n"
            f"  **Description:** {persona['description']}\n"
            f"  **Dialogue:** Generate a multi-turn conversation featuring {persona['name']} responding to a unique scenario. "
            f"  Ensure natural back-and-forth exchanges.\n\n"
        )

    return Task(
        content=task_content,
        id=str(batch),
    )
