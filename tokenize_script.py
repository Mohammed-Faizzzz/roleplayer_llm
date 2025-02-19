from datasets import load_dataset

def tokenize_function(examples):
    conversation_texts = []

    for conversation in examples:
        conversation_text = ""
        
        # Ensure each entry is properly structured with role-based prefixes
        for message in conversation:
            if message["role"] == "system":
                role_prefix = "[SYSTEM] "
            elif message["role"] == "assistant":
                role_prefix = "[ASSISTANT] "
            elif message["role"] == "user":
                role_prefix = "[USER] "
            else:
                role_prefix = "[UNKNOWN] "

            conversation_text += role_prefix + message["content"] + " "

        conversation_texts.append(conversation_text.strip())

    # Tokenize the concatenated conversation texts
    tokenized_inputs = tokenizer(
        conversation_texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Labels should be the same as input_ids (for language models)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

    return tokenized_inputs

# Load dataset
dataset = load_dataset("json", data_files="output.json")

# Check the dataset structure
print(dataset)  # This will show all available splits
print(type(dataset))  # This should print <class 'datasets.dataset_dict.DatasetDict'>

# Extract the 'train' split (or whatever is available)
dataset = dataset["train"]  # Select the train split explicitly

# Now the dataset should be a regular Hugging Face dataset, not a DatasetDict
print(type(dataset))  # Should now be <class 'datasets.arrow_dataset.Dataset'>
print(dataset[0])  # See what the first entry actually looks like

# Convert the dataset to the correct format: list of messages
def format_dialogue(examples):
    # Make sure that each entry is a list of messages
    formatted_data = []
    for key in examples:
        conversation = []
        print("key: ", key)
        print("examples[key]: ", examples[key])
        for entry in examples[key]:  # Assuming messages alternate between user and assistant
            conversation.append({
                "role": entry["role"],  # System role should always be first
                "content": entry["content"]
            })
        formatted_data.append(conversation)

    return {"text": formatted_data}

# Apply conversation formatting
dataset = dataset.map(format_dialogue, batched=True)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save the tokenized dataset
tokenized_dataset.save_to_disk("tokenized_dataset")
