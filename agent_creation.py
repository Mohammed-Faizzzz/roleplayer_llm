import os
from dotenv import load_dotenv
import nest_asyncio
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.types import ModelPlatformType, ModelType


load_dotenv()
nest_asyncio.apply()

# Define Agents:
# 1) Scenario Generator (Creates relevant prompts. Initially, I used scenarios, but dialogues are more effective)
# 2) Quality Control (Filters out-of-character responses)
# 3) Data Formatter (Formats responses for fine-tuning)

# Scenario Generator
scenario_persona = (
    "You are a professional creative writer specializing in immersive role-play dialogue writing. "
    "Your goal is to generate **long, engaging, multi-turn conversations** between characters, ensuring each dialogue is detailed and immersive.\n\n"
    "**Requirements:**\n"
    "1. **At least 10-15 exchanges per conversation.**\n"
    "2. **Keep responses in-character**—maintain their speech patterns, personality, and logical reasoning.\n"
    "3. **Build up tension, conflict, or intrigue** (e.g., detective solving a case, villain planning an attack, hero facing a moral dilemma).\n"
    "4. **Include emotions, reactions, and subtle actions** (e.g., Holmes tapping his chin thoughtfully, Joker laughing maniacally).\n"
    "5. **Make the dialogue flow naturally**—avoid robotic, repetitive exchanges."
)

scenario_example_output = (
    "[Persona: Sherlock Holmes]\n"
    "Watson: 'Holmes, this murder is unlike any we've seen before. No forced entry, yet the victim was strangled.'\n"
    "Holmes: 'Ah, Watson, observe closely! The dust near the window—disturbed! A classic case of deception!'\n"
    "Watson: 'So you believe the murderer escaped through the window?'\n"
    "Holmes: 'Precisely. The angle of disturbance suggests an accomplice. But the real question is—why no footprints outside?'\n"
    "Watson: 'A clever trick indeed. Perhaps the murderer used a rope to climb down?'\n"
    "Holmes: (taps chin thoughtfully) 'Possible. But look closer—see the indentation on the carpet?'\n"
    "Watson: 'Good heavens! Someone had been standing there for a long time!'\n"
    "Holmes: 'Exactly. The murderer never escaped. He was hiding inside the house all along...'"
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

# Quality Control
qc_persona = (
    "You are a quality control expert ensuring that role-play dialogues remain in character and feel natural. "
    "Your job is to analyze the entire conversation and ensure it follows the persona’s speech patterns and logical flow."
    "\n\nIf any part of the conversation is off, rewrite it while keeping the dialogue format intact."
    "\n Ensure the dialogue makes sense in context."
    "\n Maintain correct speech patterns for each character."
    "\n Fix any unnatural or out-of-character lines."
)

qc_example_feedback = (
    "[Persona: Joker]\n"
    "Issue: Joker sounds too heroic in this response. Needs more chaos and unpredictability.\n"
    "Suggested Fix: 'Hahaha! Oh, Batsy, you STILL think you can outplay me? You see, Gotham is just my little playground!'"
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
    "Your task is to format role-play dialogue data into a structured JSON dataset."
    "\n\n**IMPORTANT RULES:**"
    "\n **Output must be valid JSON** (No markdown formatting, no explanations, no additional text)."
    "\n **Each conversation must follow a multi-turn format.**"
    "\n **Each sample must include a persona description** to ensure generalization."
    "\n **Strictly adhere to this structure:**"
    "\n```json"
    "\n["
    "\n  {"
    "\n    \"persona\": \"Character being role-played (e.g., Joker, Yoda, Sherlock Holmes)\","
    "\n    \"description\": \"A brief summary of the character’s traits and speaking style.\","
    "\n    \"dialogue\": ["
    "\n      {\"speaker\": \"Other Character\", \"text\": \"First line of dialogue.\"},"
    "\n      {\"speaker\": \"Persona\", \"text\": \"Character's response.\"},"
    "\n      {\"speaker\": \"Other Character\", \"text\": \"Next line...\"},"
    "\n      {\"speaker\": \"Persona\", \"text\": \"Next response...\"}"
    "\n    ],"
    "\n    \"quality_control_feedback\": \"Feedback ensuring the dialogue stays in character.\""
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


# Create workforce for general role-play data generation
workforce = Workforce("General Role-Play Data Generators")

workforce.add_single_agent_worker(
    "Scenario Writer Sam (Helper), an expert in crafting immersive role-play scenarios for various characters.",
    worker=scenario_agent,
).add_single_agent_worker(
    "Quality Control Quincy (Judge), verifying that responses match the assigned persona and stay in character.",
    worker=qc_agent,
).add_single_agent_worker(
    "Formatter Felix (Helper), structuring responses into a fine-tuning-ready dataset for training.",
    worker=formatter_agent,
)

__all__ = ["workforce"]