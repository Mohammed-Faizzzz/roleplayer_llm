import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained("./deepseek_roleplayer_lora")
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./deepseek_roleplayer_lora")

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Memory dictionary to store short-term history per session
MEMORY_WINDOW_SIZE = 5
conversation_memory = {}

def generate_response(persona, user_input, session_id="default_session"):
    
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    # Retrieve the last N exchanges
    memory_text = "\n".join([f"{m['speaker']}: {m['text']}" for m in conversation_memory[session_id][-MEMORY_WINDOW_SIZE:]])

    # Construct the model input with memory
    model_input = f"[Persona: {persona}]\n{memory_text}\nUser: {user_input}\n{persona}:"

    # Generate Response
    inputs = tokenizer(model_input, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512, temperature=0.8, top_p=0.9)

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Store the new exchange in memory; keep only the last N exchanges
    conversation_memory[session_id].append({"speaker": "User", "text": user_input})
    conversation_memory[session_id].append({"speaker": persona, "text": response_text})
    conversation_memory[session_id] = conversation_memory[session_id][-MEMORY_WINDOW_SIZE:]

    return response_text
