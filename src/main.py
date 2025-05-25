import sys
import json
import os
import requests
import csv
import re

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required directories exist
for directory in [RESULTS_DIR, GROUND_TRUTH_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model_config(config_name: str) -> dict:
    """Loads the model configuration from a JSON file."""
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded model configuration from {config_path}")
    return config

def load_benchmark_tasks() -> list[dict]:
    """Loads all benchmark tasks from JSON files in the PROMPTS_DIR."""
    tasks = []
    if not os.path.exists(PROMPTS_DIR):
        print(f"Prompts directory {PROMPTS_DIR} not found.")
        return tasks

    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith(".json"):
            task_id = os.path.splitext(filename)[0]
            file_path = os.path.join(PROMPTS_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                if not task_data.get("prompt"):
                    print(f"Skipping task {task_id}: missing 'prompt' field.")
                    continue

                task_data["task_id"] = task_id
                tasks.append(task_data)
                print(f"Loaded task '{task_id}' from {filename}")

            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading task {task_id} from {filename}: {e}")
    
    print(f"Loaded {len(tasks)} tasks total.")
    return tasks

def load_ground_truth(task_id: str) -> str | None:
    """Loads the ground truth code for a task."""
    gt_path = os.path.join(GROUND_TRUTH_DIR, f"{task_id}_gt.md")
    if os.path.exists(gt_path):
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"    Error loading ground truth for {task_id}: {e}")
    return None

def get_llm_response(prompt_text: str, model_name: str, openrouter_config: dict, session: requests.Session) -> str | None:
    """Gets a response from an LLM via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {openrouter_config['api_key']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    
    try:
        response = session.post(
            f"{openrouter_config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        response.raise_for_status()
        completion = response.json()
        
        if completion.get("choices") and len(completion["choices"]) > 0:
            content = completion["choices"][0].get("message", {}).get("content")
            if content:
                return content.strip()
        
        print("  Error: No content in LLM response.")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"  Error calling OpenRouter API: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error during LLM call: {e}")
        return None

def evaluate_code_with_llm(task_prompt: str, generated_code: str, ground_truth_code: str, 
                          evaluation_criteria: str, evaluator_model: str, 
                          evaluator_config: dict, session: requests.Session) -> dict:
    """Evaluates generated code against ground truth using an LLM."""
    
    eval_prompt = f"""You are an expert code evaluator for a data science benchmark.
Compare the generated Python code against the ground truth code for the given task.

Task: {task_prompt}

Evaluation Criteria: {evaluation_criteria}

Ground Truth Code:
---
{ground_truth_code}
---

Generated Code to Evaluate:
---
{generated_code}
---

Rate the generated code on a scale of 0-10 based on:
1. Correctness (does it solve the task?)
2. Code quality (proper imports, structure, etc.)
3. Completeness (addresses all requirements?)

After the rating, provide detailed feedback.
Your response MUST be in the format: "Score: [0-10] Feedback: [Your detailed feedback]"
- If the score is between 0 and 5, the feedback should explain the primary reasons for failure and what was critically missing or incorrect.
- If the score is between 6 and 7, the feedback should highlight specific areas for improvement to achieve a higher score.
- If the score is 8 or above, the feedback should mention the strong points of the solution.
"""

    eval_response = get_llm_response(eval_prompt, evaluator_model, evaluator_config, session)
    
    if eval_response:
        try:
            # Use regex to find Score and Feedback
            match = re.search(r"Score:\s*(\d+)\s*Feedback:\s*(.+)", eval_response, re.DOTALL)
            if match:
                score_str = match.group(1)
                details = match.group(2).strip()
                score = int(score_str)
                if 0 <= score <= 10:
                    return {"score": score, "details": details}
                else: # Score is a number but out of range
                    return {"score": "Error", "details": f"Score out of range (0-10): {score_str}. Full response: {eval_response}"}
            else:
                # Fallback: try to extract just the number if the new format isn't strictly followed
                score_str_fallback = ''.join(filter(str.isdigit, eval_response.split('.')[0].splitlines()[0])) # Try to get first number
                if score_str_fallback:
                    try:
                        score_fallback = int(score_str_fallback)
                        if 0 <= score_fallback <= 10:
                             return {"score": score_fallback, "details": f"Extracted score, but feedback format was unexpected. LLM did not follow 'Score: [0-10] Feedback: [text]'. Full response: {eval_response}"}
                    except ValueError:
                        pass # score_str_fallback was not a valid int, proceed to generic error below
                return {"score": "Error", "details": f"Invalid evaluation format. Expected 'Score: [0-10] Feedback: [text]'. Received: {eval_response}"}
        except ValueError: # For int conversion if score_str or score_str_fallback is not a number
            return {"score": "Error", "details": f"Non-numeric score or unexpected format during parsing. Full response: {eval_response}"}
        except Exception as e: # Catch any other regex or parsing errors
            return {"score": "Error", "details": f"Error parsing evaluation response: {e}. Full response: {eval_response}"}
    else:
        return {"score": "Error", "details": "Failed to get evaluation response"}

def run_benchmark(model_config_name: str, evaluator_config_name: str = "evaluator_config"):
    """Main function to run the benchmark."""
    print(f"Starting benchmark for model: {model_config_name}")
    print(f"Using evaluator: {evaluator_config_name}")
    
    try:
        # Load configurations
        model_config = load_model_config(model_config_name)
        evaluator_config = load_model_config(evaluator_config_name)
        
        model_llm = model_config.get("llm", {})
        evaluator_llm = evaluator_config.get("llm", {})
        
        # Validate configurations
        for config_name, llm_config in [(model_config_name, model_llm), (evaluator_config_name, evaluator_llm)]:
            if llm_config.get("api_type") != "openrouter":
                print(f"Error: {config_name} must use 'openrouter' api_type")
                return
            if not all([llm_config.get("api_key"), llm_config.get("base_url"), llm_config.get("model")]):
                print(f"Error: {config_name} missing required fields")
                return
            if "YOUR_API_KEY" in llm_config.get("api_key", ""):
                print(f"Error: Replace 'YOUR_API_KEY' in {config_name}.json")
                return
        
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error loading configurations: {e}")
        return

    # Load tasks
    tasks = load_benchmark_tasks()
    if not tasks:
        print("No tasks found. Exiting.")
        return

    results = []
    
    with requests.Session() as session:
        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            print(f"\nProcessing task {i+1}/{len(tasks)}: {task_id}")
            
            # Get LLM response for the task
            print(f"  Generating code...")
            generated_code = get_llm_response(
                task["prompt"], 
                model_llm["model"], 
                model_llm, 
                session
            )
            
            if not generated_code:
                print(f"  Failed to generate code for {task_id}")
                results.append({
                    "task_id": task_id,
                    "difficulty": task.get("difficulty", ""),
                    "category": task.get("category", ""),
                    "generated_code": "",
                    "evaluation_score": "Error",
                    "evaluation_details": "Failed to generate code",
                    "model_config": model_config_name,
                    "evaluator_config": evaluator_config_name
                })
                continue
            
            # Load ground truth
            ground_truth = load_ground_truth(task_id)
            if not ground_truth:
                print(f"  No ground truth found for {task_id}")
                results.append({
                    "task_id": task_id,
                    "difficulty": task.get("difficulty", ""),
                    "category": task.get("category", ""),
                    "generated_code": generated_code[:500] + "..." if len(generated_code) > 500 else generated_code,
                    "evaluation_score": "No GT",
                    "evaluation_details": "No ground truth available",
                    "model_config": model_config_name,
                    "evaluator_config": evaluator_config_name
                })
                continue
            
            # Evaluate with LLM
            print(f"  Evaluating code...")
            evaluation = evaluate_code_with_llm(
                task["prompt"],
                generated_code,
                ground_truth,
                task.get("evaluation_criteria", "General code quality and correctness"),
                evaluator_llm["model"],
                evaluator_llm,
                session
            )
            
            results.append({
                "task_id": task_id,
                "difficulty": task.get("difficulty", ""),
                "category": task.get("category", ""),
                "dataset_reference": task.get("dataset_reference", ""), # ADDED
                "generated_code": generated_code[:500] + "..." if len(generated_code) > 500 else generated_code,
                "evaluation_score": evaluation["score"],
                "evaluation_details": evaluation["details"],
                "model_config": model_config_name,
                "evaluator_config": evaluator_config_name
            })
            
            print(f"  Score: {evaluation['score']}")

    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, f"{model_config_name}_results.csv")
    fieldnames = ["task_id", "difficulty", "category", "dataset_reference", "generated_code", "evaluation_score", # ADDED dataset_reference
                  "evaluation_details", "model_config", "evaluator_config"]
    
    try:
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {results_file}")
    except IOError as e:
        print(f"Error writing results: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <model_config_name> [evaluator_config_name]")
        sys.exit(1)
    
    model_config = sys.argv[1]
    eval_config = sys.argv[2] if len(sys.argv) > 2 else "evaluator_config"
    
    run_benchmark(model_config, eval_config)
