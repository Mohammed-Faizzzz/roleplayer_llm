# from agent_creation import workforce
# from data_processing import save_progress, clean_json_output
# from generate_task import generate_task
# import multiprocessing
# import time

# def process_single_batch(batch):
#     """Processes a single batch of dialogue generation."""
#     print(f"Generating batch {batch + 1}...")
#     task = generate_task(batch)
#     try:
#         task = workforce.process_task(task)
#         generated_data = task.result  # Get the generated dialogue
#         cleaned_data = clean_json_output(generated_data)  # Clean JSON output
#         if cleaned_data:
#             save_progress(cleaned_data)  # Save to file
#             print(f"Batch {batch + 1} completed and saved.")
#         else:
#             print(f"Batch {batch + 1} failed due to invalid JSON.")
#     except Exception as e:
#         print(f"Error in batch {batch + 1}: {e}")

# def generate_data_parallel(num_batches=1000, num_workers=8):
#     """Generates data in parallel using multiprocessing."""
#     print(f"Starting parallel generation with {num_workers} workers...")
#     start_time = time.time()

#     with multiprocessing.Pool(num_workers) as pool:
#         pool.map(process_single_batch, range(num_batches))

#     print(f"Data generation complete! Total time: {time.time() - start_time:.2f} seconds.")

# if __name__ == "__main__":
#     generate_data_parallel()

from agent_creation import workforce
from data_processing import save_progress, clean_json_output
from generate_task import generate_task
from persona_manager import available_personas
import time

    
def generate_data_until_budget_exhausted():
    """Generate data in batches until API credits run out"""

    dataset = []
    batch_size = 3  # Number of personas per batch
    max_batches = 50  # Target batches (stops when API credits are exhausted)

    for batch in range(max_batches):
        print(f"\nGenerating batch {batch + 1}/{max_batches}...")

        task = generate_task(batch, batch_size)  # Generate a new task

        try:
            task = workforce.process_task(task)
            generated_data = task.result  # Store generated data

            if not generated_data:
                # print("⚠️ Formatter Felix returned an empty response! Skipping batch...")
                continue

            cleaned_data = clean_json_output(generated_data)

            if cleaned_data is None:
                print("❌ Skipping batch due to invalid JSON.")
                continue

            # print("cleaned_data: ", cleaned_data)

            # **Filter for duplicates before saving**
            dataset_texts = [conv["dialogue"] for conv in dataset]  # Extract all dialogues
            new_conversations = []

            for conv in generated_data:
                text_representation = " ".join([turn["text"] for turn in conv["dialogue"]])  # Convert to raw text
                if not is_duplicate(text_representation, dataset_texts):
                    new_conversations.append(conv)
                else:
                    print(f"Duplicate detected! Skipping conversation for {conv['persona']}.")

            if new_conversations:
                dataset.extend(new_conversations)
                save_progress(new_conversations)  # Save dataset incrementally

            print(f"Valid JSON received. Saving batch {batch + 1}...")

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)

    print("\nData generation complete! Check 'roleplay_dataset.json' for results.")

if __name__ == "__main__":
    generate_data_until_budget_exhausted()
    