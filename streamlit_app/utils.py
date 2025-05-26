import pandas as pd
import os
import glob
import re

# Configuration for pricing (cost for 180 tasks)
PRICING_CONFIG = {
    'claude-3.5-sonnet' : 1.552,
    'claude-sonnet-4' : 6.640,
    'command-r7b-12-2024' : 0.014,
    'deepseek-v3': 0.189,
    'gemini-1.5-flash' : 0.025,
    'gemini-2.0-flash' : 0.062,
    'gemini-2.5-flash-preview': 0.136,
    'gemini-2.5-flash-preview-thinking': 2.108,
    'gemini-2.5-pro-preview' : 10.81,
    'gemma-3-12b-it' : 0.007,
    'gemma-3-27b-it' : 0.027,
    'mistral-nemo' : 0.007,
    'mistral-small-24b-instruct' : 0.018,
    'gpt-3.5-turbo' : 0.106,
    'gpt-4.1' : 0.994,
    'gpt-4.1-nano' : 0.043,
    'gpt-4o': 1.502,
    'gpt-4o-mini': 0.051,
    'grok-3-mini-beta' : 0.140,
    'llama-3.3-8b-instruct' : 0.0035,
    'llama-3.3-70b-instruct' : 0.037,
    'llama-4-maverick' : 0.064,
    'o4-mini' : 0.939,
}

def clean_evaluation_score(score):
    """
    Clean and validate evaluation scores, handling malformed data.
    Replace any malformed data with 0 (failed task).
    """
    if pd.isna(score):
        return 0.0
    score_str = str(score).strip()
    if not score_str:
        return 0.0
    try:
        parsed_score = float(score_str)
        return max(0.0, min(10.0, parsed_score))
    except ValueError:
        match = re.search(r'^(\\d+\\.?\\d*)', score_str)
        if match:
            try:
                parsed_score = float(match.group(1))
                return max(0.0, min(10.0, parsed_score))
            except ValueError:
                pass
        print(f"⚠️  Malformed score '{score_str[:50]}...' replaced with 0 in utils.py")
        return 0.0

def get_category_group_mapping_and_list(original_categories_set):
    """
    Maps original categories to broader groups and returns the mapping and sorted group names.
    """
    category_map = {
        "Data Loading/Understanding": ["load", "read", "import", "describe", "info", "head", "shape", "dataset overview", "data type", "column name", "file handling", "check data", "display data", "view data", "sample data"],
        "Data Cleaning/Preprocessing": ["clean", "preprocess", "missing", "null", "duplicate", "outlier", "encode", "scale", "normalize", "impute", "type conversion", "data cleansing", "handle errors", "data preparation", "data wrangling", "fill na", "drop na"],
        "Data Transformation/Feature Engineering": ["transform", "feature engineer", "create column", "derive feature", "aggregate", "group by", "merge", "join", "pivot", "feature creation", "data manipulation", "reshape data", "feature extraction", "add feature"],
        "Exploratory Data Analysis (EDA)": ["eda", "explore", "analysis", "statistic", "correlation", "distribution", "summary statistic", "data exploration", "value counts", "descriptive statistics", "investigate data", "understand data"],
        "Visualization": ["plot", "chart", "visualize", "graph", "histogram", "scatter", "bar", "heatmap", "seaborn", "matplotlib", "line plot", "box plot", "data visualization", "create plot"],
        
        "Clustering Tasks": ["cluster", "segmentation", "k-means", "kmeans", "dbscan", "hierarchical clustering", "customer segmentation", "user segmentation", "image segmentation", "unsupervised learning - clustering"],
        "Regression Tasks": ["regression", "predict continuous", "linear regression", "polynomial regression", "svr", "predict value", "forecast value", "supervised learning - regression"],
        "Classification Tasks": ["classification", "predict categorical", "logistic regression", "svm", "decision tree", "random forest", "knn", "naive bayes", "predict class", "predict label", "supervised learning - classification"],
        "Time Series Analysis": ["time series", "temporal", "forecast", "seasonality", "arima", "sarima", "prophet", "time-series analysis", "time series forecasting"],
        "Text Processing/NLP": ["text", "nlp", "string", "tokenize", "vectorize", "sentiment", "ner", "topic model", "text analysis", "natural language processing", "text mining"],
        
        "Model Building/Training (General)": ["model", "train", "fit", "algorithm", "machine learning", "deep learning", "neural network", "build model", "develop model"],
        "Model Evaluation": ["evaluate", "metric", "score", "accuracy", "precision", "recall", "f1", "roc", "auc", "confusion matrix", "performance", "cross-validation", "model assessment", "test model", "validate model"],
        
        "Data Export/Saving": ["save", "export", "write", "output", "store results", "file output", "save data", "save model"],
        "Other": [] # Catch-all for less common or very specific tasks
    }

    # Reverse map for easier lookup: keyword -> group
    # Keywords should be lowercase for case-insensitive matching
    keyword_to_group = {}
    for group, keywords in category_map.items():
        for keyword in keywords:
            keyword_to_group[keyword.lower()] = group

    grouped_categories = {group: [] for group in category_map.keys()} # group_name -> list of original_categories
    original_to_group_map = {} # original_category_str -> group_name

    unmapped_to_specific_other = []

    for cat_original in original_categories_set:
        if pd.isna(cat_original):
            continue
        
        cat_lower = str(cat_original).lower()
        mapped_to_group = None
        
        # Attempt to match with more specific keywords first if there's an implicit order
        # The current keyword_to_group structure means the last group in category_map defining a shared keyword wins.
        # The matching loop iterates through keyword_to_group, whose order depends on insertion.
        # For simplicity, we rely on distinct keywords or the first match in the keyword_to_group dict.

        best_match_group = None
        # Simple first match strategy:
        for keyword, group_name in keyword_to_group.items():
            if keyword in cat_lower:
                best_match_group = group_name
                break # Found the first keyword match

        if best_match_group:
            grouped_categories[best_match_group].append(cat_original)
            original_to_group_map[cat_original] = best_match_group
        else:
            # If no keyword matched, assign to "Other"
            grouped_categories["Other"].append(cat_original)
            original_to_group_map[cat_original] = "Other"
            if cat_original not in unmapped_to_specific_other:
                 unmapped_to_specific_other.append(cat_original)

    # Filter out groups that have no original categories mapped to them, except "Other" if it's empty
    final_grouped_categories = {g: cats for g, cats in grouped_categories.items() if cats}
    if not grouped_categories.get("Other"): # Ensure "Other" is present even if empty initially
        final_grouped_categories["Other"] = []

    sorted_group_names = sorted(final_grouped_categories.keys())

    if unmapped_to_specific_other:
        print(f"INFO (utils.py): The following original categories were mapped to 'Other' due to no specific keyword match: {unmapped_to_specific_other}")
        
    return final_grouped_categories, sorted_group_names, original_to_group_map


def load_all_results(results_dir='../results'):
    """
    Load all CSV result files from the results directory.
    Returns a dictionary with model names as keys and DataFrames as values.
    Adjusted for Streamlit context: results_dir is relative to the workspace root.
    """
    results = {}
    # Construct the absolute path to the results directory
    # The script is in streamlit_app, so ../results is correct relative to workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__)) # streamlit_app directory
    actual_results_dir = os.path.join(script_dir, results_dir) # c:\...\streamlit_app\..\results -> c:\...\results

    csv_files = glob.glob(os.path.join(actual_results_dir, '*_results.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {actual_results_dir}")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        model_name = filename.replace('_results.csv', '')
        try:
            df = pd.read_csv(file_path)
            if 'evaluation_score' in df.columns:
                df['evaluation_score'] = df['evaluation_score'].apply(clean_evaluation_score)
            
            required_columns = ['task_id', 'evaluation_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"   ⚠️  Skipping {model_name}: Missing required columns: {missing_columns}")
                continue
            
            results[model_name] = df
        except Exception as e:
            print(f"✗ Error loading {filename} in utils.py: {e}")
    return results

def calculate_model_stats(model_name, df, pricing_config):
    """
    Calculate statistics for a single model with error handling.
    """
    try:
        total_tasks = len(df)
        if 'evaluation_score' not in df.columns:
            return None
        
        scores = pd.to_numeric(df['evaluation_score'], errors='coerce')
        valid_scores = scores.dropna()
        
        if len(valid_scores) == 0:
            return None
        
        avg_score = valid_scores.mean()
        min_score = valid_scores.min()
        max_score = valid_scores.max()
        
        price_total = pricing_config.get(model_name, 0) # Cost for 180 tasks
        # If the df doesn't have 180 tasks, this price_total is for a hypothetical 180 tasks.
        # For consistency with notebook, we use this total_cost directly.
        # cost_per_task in this context would be price_total / 180 if we want to normalize
        # but the notebook used price_total / actual_total_tasks.
        # Let's stick to price_total as the 'total_cost' for the 180 task benchmark.
        
        # score_per_dollar: sum of scores / total_cost for 180 tasks
        # This assumes the model *would* process 180 tasks with similar performance.
        # If df has fewer tasks, we scale the sum of scores.
        # However, summary_df in notebook uses valid_scores.sum() / price_total.
        # Let's keep it consistent:
        score_per_dollar = valid_scores.sum() / price_total if price_total > 0 else 0
        
        # Cost per task based on actual tasks processed vs the benchmark cost
        # This might be confusing if total_tasks != 180.
        # The notebook used: price_total / total_tasks
        cost_per_task_actual = price_total / total_tasks if total_tasks > 0 else 0

        # Calculate success, incomplete, and fail counts
        successful_count = df[df['evaluation_score'] >= 8].shape[0]
        incomplete_count = df[(df['evaluation_score'] >= 6) & (df['evaluation_score'] <= 7)].shape[0]
        failed_count = df[df['evaluation_score'] <= 5].shape[0]

        # Calculate rates
        success_rate = (successful_count / total_tasks) * 100 if total_tasks > 0 else 0
        fail_incomplete_rate = ((failed_count + incomplete_count) / total_tasks) * 100 if total_tasks > 0 else 0

        return {
            'model': model_name,
            'total_tasks_processed': total_tasks, # Actual tasks in the CSV
            'valid_tasks_processed': len(valid_scores),
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'total_cost_benchmark': price_total, # Cost for 180 tasks as per PRICING_CONFIG
            'cost_per_task_actual': cost_per_task_actual, # Based on actual tasks processed
            'score_per_dollar_benchmark': score_per_dollar, # Based on benchmark cost
            'success_rate_percent': success_rate,
            'fail_incomplete_rate_percent': fail_incomplete_rate
        }
    except Exception as e:
        print(f"⚠️  Error calculating stats for {model_name} in utils.py: {e}")
        return None
