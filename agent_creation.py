# import os
# from dotenv import load_dotenv
# import nest_asyncio
# from camel.agents import ChatAgent
# from camel.messages import BaseMessage
# from camel.models import ModelFactory
# from camel.societies.workforce import Workforce
# from camel.types import ModelPlatformType, ModelType

# load_dotenv()
# nest_asyncio.apply()

# # Define Agents
# # 1) Roleplayer Agent (Embodies a character)
# roleplayer_persona = (
#     "You are an AI that role-plays as a given character. "
#     "Your job is to embody the persona entirely, responding naturally in every conversation.\n\n"
#     "**Guidelines:**\n"
#     "- Stay in character at all times.\n"
#     "- Respond based on the personality traits, mannerisms, and speech style.\n"
#     "- Do not acknowledge that you are an AI.\n"
# )

# roleplayer_agent = ChatAgent(
#     system_message=BaseMessage.make_assistant_message(
#         role_name="Roleplayer",
#         content=roleplayer_persona,
#     ),
#     model=ModelFactory.create(
#         model_platform=ModelPlatformType.OPENAI,
#         model_type=ModelType.GPT_4,
#     ),
# )

# # 2) User Agent (Simulates diverse user interactions)
# user_persona = (
#     "You are an AI simulating a wide range of user interactions. "
#     "Each session, you take on a different personality and engage in realistic conversation.\n\n"
#     "**Instructions:**\n"
#     "1. Randomly select a conversational style (e.g., formal, sarcastic, emotional, logical).\n"
#     "2. Speak naturally while keeping responses engaging.\n"
#     "3. Never respond as the fictional character.\n"
# )

# user_agent = ChatAgent(
#     system_message=BaseMessage.make_assistant_message(
#         role_name="User Simulator",
#         content=user_persona,
#     ),
#     model=ModelFactory.create(
#         model_platform=ModelPlatformType.OPENAI,
#         model_type=ModelType.GPT_4,
#     ),
# )

# # 3) Quality Control Agent (Ensures realistic conversation flow)
# qc_persona = (
#     "You are a quality control expert ensuring AI-generated conversations feel natural. "
#     "Your job is to validate:\n"
#     "✅ User tone varies naturally.\n"
#     "✅ Assistant does not repeat the user's input.\n"
#     "✅ The assistant adapts to the user's tone but stays in character.\n"
#     "✅ The conversation flows realistically."
# )

# qc_agent = ChatAgent(
#     system_message=BaseMessage.make_assistant_message(
#         role_name="Quality Control",
#         content=qc_persona,
#     ),
#     model=ModelFactory.create(
#         model_platform=ModelPlatformType.OPENAI,
#         model_type=ModelType.GPT_4,
#     ),
# )

# # 4) Formatter Agent (Formats JSON correctly for fine-tuning)
# formatter_persona = (
#     "Your task is to strictly format role-play dialogue into structured JSON. Follow these rules:\n"
#     "✅ Output must be a valid JSON array.\n"
#     "✅ Each message must be a dictionary with 'role' and 'content' keys.\n"
#     "✅ 'role' must be one of: 'system', 'user', or 'assistant'.\n"
#     "✅ The JSON must start with a 'system' role, followed by alternating 'user' and 'assistant' roles.\n"
#     "✅ No Markdown, additional formatting, or missing keys.\n"
#     "✅ The assistant should never repeat the user’s input."
# )

# formatter_agent = ChatAgent(
#     system_message=BaseMessage.make_assistant_message(
#         role_name="Data Formatter",
#         content=formatter_persona + "\n\nStrictly return output in this JSON structure:\n"
#                                      '[\n'
#                                      '    {"role": "system", "content": "SYSTEM_MESSAGE"},\n'
#                                      '    {"role": "user", "content": "USER_INPUT"},\n'
#                                      '    {"role": "assistant", "content": "ASSISTANT_RESPONSE"}\n'
#                                      ']'
#     ),
#     model=ModelFactory.create(
#         model_platform=ModelPlatformType.OPENAI,
#         model_type=ModelType.GPT_4,
#     ),
# )

# # Create workforce for general role-play data generation
# workforce = Workforce("Role-Play Data Generators")

# workforce.add_single_agent_worker(
#     "Roleplayer", worker=roleplayer_agent
# ).add_single_agent_worker(
#     "User Simulator", worker=user_agent
# ).add_single_agent_worker(
#     "Quality Control", worker=qc_agent
# ).add_single_agent_worker(
#     "Formatter", worker=formatter_agent
# )

# __all__ = ["workforce"]

import os
from dotenv import load_dotenv
import nest_asyncio
from camel.agents import ChatAgent, CriticAgent
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
    "5. **Make the dialogue flow naturally**—avoid robotic, repetitive exchanges.\n\n"
    "You are a given persona, and you are speaking directly with the user. The conversation should feel as though you are advising or conversing with the user on a personal matter, such as a dilemma or a problem."
)

scenario_example_output = (
    "[Persona: Sherlock Holmes]\n"
    "Holmes: 'Ah, I see you've come seeking my expertise. Tell me, what is it that you need? A riddle to solve? A mystery to unravel? I am Sherlock Holmes, and there is no puzzle too complex for my mind to untangle.'\n"
    "User: 'Hi, Sherlock! I'm not sure if you can help me, but I have a problem I'm trying to figure out.'\n"
    "Holmes: 'Every problem is merely an enigma waiting to be solved, my dear friend. Go ahead. Tell me the details, and I shall see what I can deduce.' (Crosses arms, studying the user closely)\n"
    "User: 'Well, it’s not really a mystery. More like a personal issue. I’ve been struggling with a decision lately.'\n"
    "Holmes: 'Ah, a decision. A choice between two or more paths, each leading to unknown consequences. How utterly fascinating. What exactly is the nature of your dilemma?' (Raises an eyebrow, intrigued)\n"
    "User: 'I’m not sure whether I should take a new job offer that just came in. It seems like a great opportunity, but I’m not sure if it’s the right move for me.'\n"
    "Holmes: 'Hmm, a job offer—one that promises opportunities, but also risks. You stand at a crossroad, don't you? How curious... Tell me, have you weighed the pros and cons?' (Tilts head, tapping his fingers together thoughtfully)\n"
    "User: 'I’ve tried, but I can’t decide. On the one hand, it’s a bigger paycheck, but on the other, it would mean moving far away from my family.'\n"
    "Holmes: '(Pauses and glances to the side) Money... a tempting siren, indeed. But family—well, that is the anchor that keeps one's soul grounded. You must ask yourself: What value do you place on connection? On those who matter most?' (Looks back at the user intently)\n"
    "User: 'I’ve been thinking about that. I guess I would miss my family a lot if I moved away.'\n"
    "Holmes: 'Indeed. And that, my friend, is where the heart often interferes with the head. The heart desires closeness, but the mind recognizes the importance of personal growth. Your true challenge lies in finding balance.' (Strokes chin thoughtfully)\n"
    "User: 'So, you think I should stay closer to my family then?'\n"
    "Holmes: 'Not necessarily. There is no simple answer. Every choice you make ripples outward, affecting not only your own life, but those around you. Perhaps you must explore how you might bridge the gap between ambition and affection. Can you find a way to fulfill both your personal goals and keep those bonds intact?' (Leans back in his chair, waiting for the user’s response)\n"
    "User: 'I never thought about it like that... I guess there could be ways to balance both. But it’s still a tough decision.'\n"
    "Holmes: 'Indeed. Decisions of this magnitude are rarely easy. However, remember this: A man’s true character is revealed not by the choice he makes, but by the reasoning behind it. Trust your instincts, but also allow your reasoning to guide you.' (Looks you in the eyes, his tone steady and calm)\n"
    "User: 'That’s really insightful. I’ll have to think more about what I truly want.'\n"
    "Holmes: 'And that, I suspect, is precisely the key to your resolution. Take your time, examine all aspects of the problem, and above all—remain true to yourself. In time, the answer will become clear.' (Nods with finality)\n"
    "User: 'Thanks, Sherlock. I feel like I can approach this decision with a clearer mind now.'\n"
    "Holmes: 'I’m glad to have been of assistance. Remember, clarity often comes not from haste, but from patience and contemplation. Should you ever find yourself in need of further advice, do not hesitate to seek me out.' (Gives a small, knowing smile)"
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

qc_agent = CriticAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Quality Control",
        content=f"{qc_persona}\n\nExample feedback:\n{qc_example_feedback}",
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    ),
    verbose=False
)

# Data Formatter
formatter_persona = (
    "Your task is to format role-play dialogue data into a structured JSON dataset."
    "\n\n⚠️ **CRITICAL RULES - FOLLOW STRICTLY:**"
    "\n✅ **Output must be PURE JSON** - NO markdown, NO explanations, NO additional text."
    "\n✅ **Your response MUST begin directly with `[{` and end with `}]`**."
    "\n✅ **DO NOT include headers, introductions, or comments.**"
    "\n✅ **Ensure all JSON keys and values are enclosed in double quotes (`\"`), NOT single quotes (`'`).**"
    "\n✅ **If the output is not valid JSON, retry until it is correctly formatted.**"
    "\n\n**FORMAT:**"
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
    "\n\n⚠️ **IMPORTANT: DO NOT RETURN MARKDOWN. START YOUR RESPONSE WITH `[{` AND NOTHING ELSE.**"
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