from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.mps.empty_cache()
device = torch.device("cpu")

# Load Base LLaMA-3-3B Model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA --> Suited for our use case
lora_config = LoraConfig(
    r=16,  # Low-rank adaptation size
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to key attention layers
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.to(device)

def format_dialogue(example):
    dialogue_text = f"[Persona: {example['persona']}]\n"
    dialogue_text += f"Description: {example['description']}\n\n"
    
    for turn in example["dialogue"]:
        dialogue_text += f"{turn['speaker']}: {turn['text']}\n"

    return {"text": dialogue_text}

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",   # Ensures all sequences have the same length
        truncation=True,        # Prevents sequences from being too long
        max_length=512,         # Set max length (adjust based on model limits)
        return_tensors="pt"
    )

# Load dataset
dataset = load_dataset("json", data_files="generalised.json")
# At least 100 different personas with a total of 1000-5000 data points (ideally 100K-500K data points)
dataset = dataset.map(format_dialogue)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.save_to_disk("tokenized_dataset")
train_dataset = tokenized_dataset['train']

# Training Hyperparameters
training_args = TrainingArguments(
    output_dir="./deepseek_roleplayer_model",
    per_device_train_batch_size=1,  # Reduce batch size (try 1, 2, or 4)
    per_device_eval_batch_size=1,  # Reduce batch size for evaluation too
    gradient_accumulation_steps=8,
    num_train_epochs=3,  # Start with 3, increase if needed
    learning_rate=5e-4,  # Higher LR needed since it's a distilled model
    save_steps=500,  # Save progress frequently
    logging_steps=50,  # Monitor training
    optim="adamw_torch",  # AdamW works best for Transformers
    weight_decay=0.0,  # Distilled models don’t need weight decay
    warmup_steps=500,  # Smooth LR transition
    fp16=True,  # Mixed precision for speed
    bf16=False,
    evaluation_strategy="no",  # Monitor performance
    report_to="none",  # Disable external logging
    lr_scheduler_type="cosine",  # Decays learning rate over time
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Save the model and Tokenizer
model.save_pretrained("./deepseek_roleplayer_lora")
tokenizer.save_pretrained("./deepseek_roleplayer_lora")

print("Model and tokenizer saved successfully!")