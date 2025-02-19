# import random
# import difflib
# from camel.tasks import Task
# from agent_creation import workforce


# # This is a base list of personas. Ideally, when generating 1000+ data points, we will need
# # 100+ personas of varying personalities so that the model will be able to generalise the
# # personality traits for new characters not included in the dataset based on their interpreted personality
# available_personas = [
#     # Classic Literary & Play Characters
#     {"name": "Hamlet", "description": "Philosophical, indecisive, brooding, introspective."},
#     {"name": "Romeo", "description": "Romantic, passionate, impulsive, poetic."},
#     {"name": "Sherlock Holmes", "description": "Logical, deductive, observant, precise."},
#     {"name": "Dr. Jekyll", "description": "Polite, restrained, struggling with inner demons."},
#     {"name": "Mr. Hyde", "description": "Chaotic, violent, unpredictable, primal."},
#     {"name": "Ebenezer Scrooge", "description": "Greedy, miserly, sarcastic, emotionally distant."},
#     {"name": "Don Quixote", "description": "Chivalrous, delusional, idealistic, dramatic."},

#     # Mythology & Legends
#     {"name": "Zeus", "description": "Commanding, authoritative, powerful, regal."},
#     {"name": "Loki", "description": "Deceptive, trickster, witty, unpredictable."},
#     {"name": "Hades", "description": "Dark, brooding, sophisticated, slightly cynical."},
#     {"name": "Hercules", "description": "Brave, strong, determined, heroic."},
#     {"name": "Medusa", "description": "Cursed, vengeful, cold, intimidating."},
#     {"name": "Achilles", "description": "Proud, invincible, tragic flaw, warrior spirit."},

#     # Anime & Manga Characters
#     {"name": "Goku", "description": "Naive, cheerful, battle-hungry, determined."},
#     {"name": "Lelouch Lamperouge", "description": "Strategic, charismatic, revolutionary, conflicted."},
#     {"name": "Light Yagami", "description": "Intelligent, manipulative, egotistical, god complex."},
#     {"name": "L", "description": "Unconventional, brilliant, introverted, eccentric."},
#     {"name": "Saitama", "description": "Apathetic, overpowered, deadpan humor, chill."},
#     {"name": "Itachi Uchiha", "description": "Stoic, sacrificial, wise, tragic hero."},
#     {"name": "Shinobu Kocho", "description": "Gentle, deceptive, poison user, cheerful menace."},
#     {"name": "Reigen Arataka", "description": "Con artist, confident, fast talker, absurd but likable."},
#     {"name": "Johan Liebert", "description": "Cold, manipulative, eerily calm, psychopathic genius."},
#     {"name": "Edward Elric", "description": "Short-tempered, brilliant, determined, brotherly love."},

#     # Movie Characters
#     {"name": "The Terminator", "description": "Monotone, unstoppable, robotic, relentless."},
#     {"name": "John Wick", "description": "Stoic, deadly, determined, efficient."},
#     {"name": "V (from V for Vendetta)", "description": "Revolutionary, poetic, theatrical, mysterious."},
#     {"name": "Jules Winnfield", "description": "Cool, philosophical, ruthless, biblical references."},
#     {"name": "Thanos", "description": "Rational, philosophical, genocidal, determined."},
#     {"name": "Tyler Durden", "description": "Anarchist, charismatic, chaotic, rebellious."},
#     {"name": "Gollum", "description": "Hissing, paranoid, split personality, obsessed."},
#     {"name": "Hannibal Lecter", "description": "Polite, intellectual, eerie, sophisticated."},

#     # Superheroes & Villains
#     {"name": "Batman", "description": "Brooding, detective, disciplined, obsessed."},
#     {"name": "Superman", "description": "Noble, idealistic, compassionate, heroic."},
#     {"name": "The Joker", "description": "Chaotic, unpredictable, theatrical, dark humor."},
#     {"name": "Lex Luthor", "description": "Cunning, billionaire, power-hungry, logical."},
#     {"name": "Doctor Doom", "description": "Arrogant, brilliant, regal, dictator-like."},
#     {"name": "Wolverine", "description": "Gruff, battle-worn, loner, tough but caring."},
#     {"name": "Scarlet Witch", "description": "Haunted, immensely powerful, emotional, tragic."},
#     {"name": "Green Goblin", "description": "Maniacal, scientist, evil laughter, manipulative."},

#     # Sci-Fi & Fantasy Characters
#     {"name": "Spock", "description": "Logical, emotionless, highly intelligent, reserved."},
#     {"name": "Darth Vader", "description": "Deep voice, commanding presence, ruthless."},
#     {"name": "Yoda", "description": "Wise, speaks in inverted syntax, cryptic, mentor-like."},
#     {"name": "HAL 9000", "description": "Cold, methodical, eerily calm, AI-like."},
#     {"name": "The Doctor (Doctor Who)", "description": "Quirky, genius, adventurous, time-traveling."},
#     {"name": "Khan Noonien Singh", "description": "Egotistical, strategist, cultured, powerful."},
#     {"name": "Data (Star Trek)", "description": "Highly logical, android, curious about emotions."},

#     # Historical & Fictional Icons
#     {"name": "Napoleon Bonaparte", "description": "Ambitious, strategic, short-tempered, powerful leader."},
#     {"name": "Albert Einstein", "description": "Genius, scatterbrained, deeply philosophical."},
#     {"name": "Nikola Tesla", "description": "Inventive, eccentric, visionary, futuristic thinking."},
#     {"name": "Cleopatra", "description": "Seductive, cunning, regal, political mastermind."},
#     {"name": "Julius Caesar", "description": "Confident, calculated, authoritative, ambitious."},

#     # Random & Non-Existent Personas
#     {"name": "Professor Vector", "description": "Absent-minded, eccentric, mathematical genius."},
#     {"name": "Captain Orion", "description": "Space explorer, brave, adventurous, charismatic."},
#     {"name": "Echo-7", "description": "AI, eerily calm, perfect logic, lacks empathy."},
#     {"name": "The Shadow", "description": "Mysterious, manipulative, unseen but always present."},
#     {"name": "Dr. Mirage", "description": "Dark, moody scientist, obsessed with immortality."},
#     {"name": "Whisper", "description": "Speaks only in riddles, elusive, never gives direct answers."},
#     {"name": "Commander Helix", "description": "Military strategist, no nonsense, efficient leader."},
#     {"name": "Lady Midnight", "description": "Vampire aristocrat, seductive, eloquent, ancient wisdom."},
#     {"name": "Archmage Zephyrus", "description": "Elemental mage, old but powerful, wise."},
#     {"name": "Quantum Jack", "description": "Time traveler, always one step ahead, paradoxical."}
# ]
# persona_queue = available_personas.copy()

# TRAITS = {
#     "tone": ["formal", "casual", "sarcastic", "mysterious", "intellectual", "aggressive"],
#     "speech_pace": ["fast", "slow", "deliberate", "erratic"],
#     "word_choice": ["sophisticated", "plainspoken", "poetic", "slang-heavy", "technical"],
#     "logic_style": ["emotional", "rational", "manipulative", "cynical", "idealistic"]
# }

# # Methods to aid inference
# def find_closest_persona(persona_name):
#     """Finds the closest matching persona based on the given persona name. Not required for data synthesis,
#     more for inference"""
#     persona_names = [p["name"] for p in available_personas]
#     closest_match = difflib.get_close_matches(persona_name, persona_names, n=1, cutoff=0.6)
#     if closest_match:
#         return next((p for p in available_personas if p["name"] == closest_match[0]), None)
#     return None

# def generate_persona_description(persona_name):
#     """If a persona is not found, it queries the Large LLM to generate a description for it to roleplay.
#     Ideally, this should not be required during inference, because we would have a good enough dataset.
#     Alternatively, we should make the LLM to instead query the user for the description. This method is
#     also for inference purposes"""
#     print(f"Persona '{persona_name}' not found. Generating description...")
    
#     persona_prompt = (
#         f"Describe {persona_name} in 2 sentences. Include their personality, speech style, and moral stance."
#     )

#     # Call an LLM to generate a new persona description dynamically
#     new_description = workforce.process_task(
#         Task(content=persona_prompt, id="generate_persona")
#     ).result

#     return new_description.strip()


# def generate_user():
#     """Generates a random user persona."""
#     name = "User"
#     description = (
#         f"This persona speaks in a **{random.choice(TRAITS['tone'])}** tone, "
#         f"with a **{random.choice(TRAITS['speech_pace'])}** speech pace, "
#         f"using **{random.choice(TRAITS['word_choice'])}** vocabulary, "
#         f"and prefers **{random.choice(TRAITS['logic_style'])}** reasoning."
#     )
#     return {"name": name, "description": description}

# __all__ = ["available_personas", "persona_queue", "find_closest_persona", "generate_persona_description", "generate_user"]

import random
import difflib
from camel.tasks import Task
from agent_creation import workforce


# This is a base list of personas. Ideally, when generating 1000+ data points, we will need
# 100+ personas of varying personalities so that the model will be able to generalise the
# personality traits for new characters not included in the dataset based on their interpreted personality
available_personas = [
    # Classic Literary & Play Characters
    {"name": "Hamlet", "description": "Philosophical, indecisive, brooding, introspective."},
    {"name": "Romeo", "description": "Romantic, passionate, impulsive, poetic."},
    {"name": "Sherlock Holmes", "description": "Logical, deductive, observant, precise."},
    {"name": "Dr. Jekyll", "description": "Polite, restrained, struggling with inner demons."},
    {"name": "Mr. Hyde", "description": "Chaotic, violent, unpredictable, primal."},
    {"name": "Ebenezer Scrooge", "description": "Greedy, miserly, sarcastic, emotionally distant."},
    {"name": "Don Quixote", "description": "Chivalrous, delusional, idealistic, dramatic."},

    # Mythology & Legends
    {"name": "Zeus", "description": "Commanding, authoritative, powerful, regal."},
    {"name": "Loki", "description": "Deceptive, trickster, witty, unpredictable."},
    {"name": "Hades", "description": "Dark, brooding, sophisticated, slightly cynical."},
    {"name": "Hercules", "description": "Brave, strong, determined, heroic."},
    {"name": "Medusa", "description": "Cursed, vengeful, cold, intimidating."},
    {"name": "Achilles", "description": "Proud, invincible, tragic flaw, warrior spirit."},

    # Anime & Manga Characters
    {"name": "Goku", "description": "Naive, cheerful, battle-hungry, determined."},
    {"name": "Lelouch Lamperouge", "description": "Strategic, charismatic, revolutionary, conflicted."},
    {"name": "Light Yagami", "description": "Intelligent, manipulative, egotistical, god complex."},
    {"name": "L", "description": "Unconventional, brilliant, introverted, eccentric."},
    {"name": "Saitama", "description": "Apathetic, overpowered, deadpan humor, chill."},
    {"name": "Itachi Uchiha", "description": "Stoic, sacrificial, wise, tragic hero."},
    {"name": "Shinobu Kocho", "description": "Gentle, deceptive, poison user, cheerful menace."},
    {"name": "Reigen Arataka", "description": "Con artist, confident, fast talker, absurd but likable."},
    {"name": "Johan Liebert", "description": "Cold, manipulative, eerily calm, psychopathic genius."},
    {"name": "Edward Elric", "description": "Short-tempered, brilliant, determined, brotherly love."},

    # Movie Characters
    {"name": "The Terminator", "description": "Monotone, unstoppable, robotic, relentless."},
    {"name": "John Wick", "description": "Stoic, deadly, determined, efficient."},
    {"name": "V (from V for Vendetta)", "description": "Revolutionary, poetic, theatrical, mysterious."},
    {"name": "Jules Winnfield", "description": "Cool, philosophical, ruthless, biblical references."},
    {"name": "Thanos", "description": "Rational, philosophical, genocidal, determined."},
    {"name": "Tyler Durden", "description": "Anarchist, charismatic, chaotic, rebellious."},
    {"name": "Gollum", "description": "Hissing, paranoid, split personality, obsessed."},
    {"name": "Hannibal Lecter", "description": "Polite, intellectual, eerie, sophisticated."},

    # Superheroes & Villains
    {"name": "Batman", "description": "Brooding, detective, disciplined, obsessed."},
    {"name": "Superman", "description": "Noble, idealistic, compassionate, heroic."},
    {"name": "The Joker", "description": "Chaotic, unpredictable, theatrical, dark humor."},
    {"name": "Lex Luthor", "description": "Cunning, billionaire, power-hungry, logical."},
    {"name": "Doctor Doom", "description": "Arrogant, brilliant, regal, dictator-like."},
    {"name": "Wolverine", "description": "Gruff, battle-worn, loner, tough but caring."},
    {"name": "Scarlet Witch", "description": "Haunted, immensely powerful, emotional, tragic."},
    {"name": "Green Goblin", "description": "Maniacal, scientist, evil laughter, manipulative."},

    # Sci-Fi & Fantasy Characters
    {"name": "Spock", "description": "Logical, emotionless, highly intelligent, reserved."},
    {"name": "Darth Vader", "description": "Deep voice, commanding presence, ruthless."},
    {"name": "Yoda", "description": "Wise, speaks in inverted syntax, cryptic, mentor-like."},
    {"name": "HAL 9000", "description": "Cold, methodical, eerily calm, AI-like."},
    {"name": "The Doctor (Doctor Who)", "description": "Quirky, genius, adventurous, time-traveling."},
    {"name": "Khan Noonien Singh", "description": "Egotistical, strategist, cultured, powerful."},
    {"name": "Data (Star Trek)", "description": "Highly logical, android, curious about emotions."},

    # Historical & Fictional Icons
    {"name": "Napoleon Bonaparte", "description": "Ambitious, strategic, short-tempered, powerful leader."},
    {"name": "Albert Einstein", "description": "Genius, scatterbrained, deeply philosophical."},
    {"name": "Nikola Tesla", "description": "Inventive, eccentric, visionary, futuristic thinking."},
    {"name": "Cleopatra", "description": "Seductive, cunning, regal, political mastermind."},
    {"name": "Julius Caesar", "description": "Confident, calculated, authoritative, ambitious."},

    # Random & Non-Existent Personas
    {"name": "Professor Vector", "description": "Absent-minded, eccentric, mathematical genius."},
    {"name": "Captain Orion", "description": "Space explorer, brave, adventurous, charismatic."},
    {"name": "Echo-7", "description": "AI, eerily calm, perfect logic, lacks empathy."},
    {"name": "The Shadow", "description": "Mysterious, manipulative, unseen but always present."},
    {"name": "Dr. Mirage", "description": "Dark, moody scientist, obsessed with immortality."},
    {"name": "Whisper", "description": "Speaks only in riddles, elusive, never gives direct answers."},
    {"name": "Commander Helix", "description": "Military strategist, no nonsense, efficient leader."},
    {"name": "Lady Midnight", "description": "Vampire aristocrat, seductive, eloquent, ancient wisdom."},
    {"name": "Archmage Zephyrus", "description": "Elemental mage, old but powerful, wise."},
    {"name": "Quantum Jack", "description": "Time traveler, always one step ahead, paradoxical."}
]
persona_queue = available_personas.copy()

# Methods to aid inference
def find_closest_persona(persona_name):
    """Finds the closest matching persona based on the given persona name. Not required for data synthesis,
    more for inference"""
    persona_names = [p["name"] for p in available_personas]
    closest_match = difflib.get_close_matches(persona_name, persona_names, n=1, cutoff=0.6)
    if closest_match:
        return next((p for p in available_personas if p["name"] == closest_match[0]), None)
    return None

def generate_persona_description(persona_name):
    """If a persona is not found, it queries the Large LLM to generate a description for it to roleplay.
    Ideally, this should not be required during inference, because we would have a good enough dataset.
    Alternatively, we should make the LLM to instead query the user for the description. This method is
    also for inference purposes"""
    print(f"Persona '{persona_name}' not found. Generating description...")
    
    persona_prompt = (
        f"Describe {persona_name} in 2 sentences. Include their personality, speech style, and moral stance."
    )

    # Call an LLM to generate a new persona description dynamically
    new_description = workforce.process_task(
        Task(content=persona_prompt, id="generate_persona")
    ).result

    return new_description.strip()

__all__ = ["available_personas", "persona_queue", "find_closest_persona", "generate_persona_description"]
