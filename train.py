from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load Base LLaMA-3-3B Model
model_name = "deepseek-ai/deepseek-llm-1.5b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA --> Suited for our use case
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # LoRA for language models
    r=8,  # LoRA rank --> (lower = less compute, higher = more adaptation)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
)

model = get_peft_model(model, lora_config)

# Load dataset
train_dataset = load_dataset("json", data_files="generalised.json")
# At least 100 different personas with a total of 1000-5000 data points (ideally 100K-500K data points)

# Training Hyperparameters
training_args = TrainingArguments(
    output_dir="./roleplayer_model",
    per_device_train_batch_size=4, # small size to save memory
    gradient_accumulation_steps=16, # simulate larger batch
    num_train_epochs=3, # given small dataset size, only a low num of epochs reqd
    learning_rate=3e-4,  # Higher LR for LoRA bc LoRA adapts quickly
    save_steps=500,
    logging_steps=50,
    optim="adamw_torch",
    warmup_steps=500, # gradually increase lr at the start to avoid sudden weight updates
    fp16=True,  # Mixed precision training will be faster and occupy less memory
    report_to="none",  # Disable logging to wandb/huggingface
    evaluation_strategy="steps",
    eval_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train LoRA Model
trainer.train()
