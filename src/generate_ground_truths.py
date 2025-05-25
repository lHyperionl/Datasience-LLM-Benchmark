import sys
import json
import os
import requests
import time

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truth")

# Ensure ground_truth directory exists
if not os.path.exists(GROUND_TRUTH_DIR):
    os.makedirs(GROUND_TRUTH_DIR)
    print(f"Created directory: {GROUND_TRUTH_DIR}")

def load_llm_config(config_name: str) -> dict:
    """Loads the LLM configuration from a JSON file."""
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"LLM configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded LLM configuration from {config_path}")
    return config

def ground_truth_exists(task_id: str) -> bool:
    """Check if ground truth file already exists for a task."""
    gt_filepath = os.path.join(GROUND_TRUTH_DIR, f"{task_id}_gt.md")
    return os.path.exists(gt_filepath)

def load_tasks_for_gt() -> list[dict]:
    """Loads all benchmark tasks from JSON files in the PROMPTS_DIR for GT generation.
    Only loads tasks that don't already have ground truth files."""
    tasks = []
    if not os.path.exists(PROMPTS_DIR):
        print(f"Prompts directory {PROMPTS_DIR} not found.")
        return tasks

    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith(".json"):
            task_id = os.path.splitext(filename)[0]
            
            # Check if ground truth already exists
            if ground_truth_exists(task_id):
                print(f"Skipping task {task_id}: ground truth already exists")
                continue
            
            file_path = os.path.join(PROMPTS_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                if not task_data.get("prompt"):
                    print(f"Skipping task {task_id}: missing 'prompt' field.")
                    continue

                task_data["task_id"] = task_id
                tasks.append(task_data)
                print(f"Loaded task '{task_id}' for GT generation")

            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading task {task_id}: {e}")
    
    print(f"Loaded {len(tasks)} tasks for GT generation")
    return tasks

def generate_gt_prompt(task_data: dict) -> str:
    """Generates the prompt to send to the LLM for ground truth generation."""
    prompt = task_data.get("prompt")
    dataset_reference = task_data.get("dataset_reference", "")
    evaluation_criteria = task_data.get("evaluation_criteria", "")
    
    gt_prompt = f"""You are an expert data scientist. Generate a comprehensive Python code solution for the following task.

Task Prompt:
---
{prompt}
---
"""
    
    if dataset_reference:
        gt_prompt += f"\nDataset Reference: {dataset_reference}\n"
    
    if evaluation_criteria:
        gt_prompt += f"\nEvaluation Criteria: {evaluation_criteria}\n"
    
    gt_prompt += """
Requirements for the solution:
1. Provide complete, working Python code
2. Include all necessary imports
3. Add clear comments explaining each step
4. Handle edge cases appropriately
5. Follow best practices for data science code

Provide ONLY the Python code with comments, no additional explanations."""
    
    return gt_prompt

def get_llm_response(prompt_text: str, llm_config: dict, session: requests.Session) -> str | None:
    """Gets a response from an LLM via OpenRouter for GT generation."""
    openrouter_cfg = llm_config.get("llm")
    if not openrouter_cfg or openrouter_cfg.get("api_type") != "openrouter":
        print("  Error: LLM configuration must be for 'openrouter'.")
        return None

    model_name = openrouter_cfg.get("model")
    api_key = openrouter_cfg.get("api_key")
    base_url = openrouter_cfg.get("base_url")

    if not all([model_name, api_key, base_url]):
        print("  Error: OpenRouter config missing model, api_key, or base_url.")
        return None
    if "YOUR_API_KEY" in api_key:
        print("  Error: Please replace 'YOUR_API_KEY' in your LLM config file.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.1  # Low temperature for consistent, high-quality code
    }
    
    print(f"    Querying LLM: {model_name}")
    try:
        response = session.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=300
        )
        response.raise_for_status()
        completion = response.json()
        
        if completion.get("choices") and len(completion["choices"]) > 0:
            content = completion["choices"][0].get("message", {}).get("content")
            if content:
                return content.strip()
        
        print("    Error: No content in LLM response.")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"    Error calling OpenRouter API: {e}")
        return None
    except Exception as e:
        print(f"    Unexpected error during LLM call: {e}")
        return None

def save_ground_truth(task_id: str, gt_content: str):
    """Saves the generated ground truth content to a file."""
    gt_filename = f"{task_id}_gt.md"
    gt_filepath = os.path.join(GROUND_TRUTH_DIR, gt_filename)
    try:
        with open(gt_filepath, 'w', encoding='utf-8') as f:
            f.write(gt_content)
        print(f"    Successfully saved ground truth to: {gt_filepath}")
    except IOError as e:
        print(f"    Error saving ground truth file {gt_filepath}: {e}")

def generate_ground_truths(llm_config_name: str):
    """Main function to generate ground truths for all tasks."""
    print(f"Starting ground truth generation using LLM config: {llm_config_name}")
    
    try:
        llm_config = load_llm_config(llm_config_name)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error loading LLM configuration: {e}")
        return

    tasks = load_tasks_for_gt()
    if not tasks:
        print("No tasks found to generate ground truths for. Exiting.")
        return

    with requests.Session() as session:
        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            print(f"\nProcessing task {i+1}/{len(tasks)}: {task_id}")

            # Generate prompt for ground truth
            gt_prompt = generate_gt_prompt(task)
            
            # Query LLM for ground truth
            print(f"  Generating ground truth for {task_id}...")
            ground_truth_response = get_llm_response(gt_prompt, llm_config, session)

            if ground_truth_response:
                save_ground_truth(task_id, ground_truth_response)
            else:
                print(f"  Failed to generate ground truth for task {task_id}.")
            
            time.sleep(2)  # Be respectful to the API

    print("\nGround truth generation process completed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_ground_truths_new.py <llm_config_name>")
        print("Example: python generate_ground_truths_new.py evaluator_config")
        sys.exit(1)
    
    llm_config_name = sys.argv[1]
    generate_ground_truths(llm_config_name)
