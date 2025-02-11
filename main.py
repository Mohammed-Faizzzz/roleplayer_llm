from agent_creation import workforce
from data_processing import save_progress, clean_json_output
from generate_task import generate_task
from persona_manager import available_personas

    
def generate_data_until_budget_exhausted():
    """Generate data in batches until API credits run out"""

    dataset = []
    batch_size = 5  # Number of personas per batch
    max_batches = 50  # Target batches (stops when API credits are exhausted)

    for batch in range(max_batches):
        print(f"\nGenerating batch {batch + 1}/{max_batches}...")

        task = generate_task(batch, batch_size)  # Use the new persona generator

        try:
            task = workforce.process_task(task)
            generated_data = task.result  # Store generated data
            
            # Validate JSON before saving
            cleaned_data = clean_json_output(generated_data)
            if cleaned_data is None:
                continue  # Skip this batch if JSON is invalid
            
            print(f"Valid JSON received. Saving batch {batch + 1}...")
            dataset.extend(cleaned_data)

            save_progress(cleaned_data)  # Save dataset incrementally

        except (openai.RateLimitError, openai.InvalidRequestError) as e:
            print(f"\nOpenAI API limit reached! ({e})")
            print("Budget likely exhausted. Stopping execution and saving progress.")
            break  # Exit loop safely

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)

    print("\nData generation complete! Check 'roleplay_dataset.json' for results.")

if __name__ == "__main__":
    generate_data_until_budget_exhausted()