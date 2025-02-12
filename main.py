from agent_creation import workforce
from data_processing import save_progress, clean_json_output
from generate_task import generate_task
from persona_manager import available_personas

    
def generate_data_until_budget_exhausted():
    """Generate data in batches until API credits run out"""

    dataset = []
    batch_size = 3  # Number of personas per batch
    max_batches = 50  # Target batches (stops when API credits are exhausted)

    for batch in range(max_batches):
        print(f"\n🚀 Generating batch {batch + 1}/{max_batches}...")

        task = generate_task(batch, batch_size)  # Generate a new task

        try:
            task = workforce.process_task(task)
            generated_data = task.result  # Store generated data

            # Validate JSON before saving
            cleaned_data = clean_json_output(generated_data)
            if cleaned_data is None:
                continue  # Skip this batch if JSON is invalid

            # **Filter for duplicates before saving**
            dataset_texts = [conv["dialogue"] for conv in dataset]  # Extract all dialogues
            new_conversations = []

            for conv in cleaned_data:
                text_representation = " ".join([turn["text"] for turn in conv["dialogue"]])  # Convert to raw text
                if not is_duplicate(text_representation, dataset_texts):
                    new_conversations.append(conv)
                else:
                    print(f"⚠️ Duplicate detected! Skipping conversation for {conv['persona']}.")

            if new_conversations:
                dataset.extend(new_conversations)
                save_progress(new_conversations)  # Save dataset incrementally

            print(f"✅ Valid JSON received. Saving batch {batch + 1}...")

        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            print("⏳ Waiting 5 seconds before retrying...")
            time.sleep(5)

    print("\n✅ Data generation complete! Check 'roleplay_dataset.json' for results.")

if __name__ == "__main__":
    generate_data_until_budget_exhausted()