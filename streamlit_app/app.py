import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Added numpy import
from utils import load_all_results, calculate_model_stats, PRICING_CONFIG, get_category_group_mapping_and_list

st.set_page_config(
    layout="wide",
    page_title="Datascience LLM Benchmark Dashboard",
    page_icon="🚀"
)

# Load data using the utility function
all_results_data_original = load_all_results(results_dir='../results')

if not all_results_data_original:
    st.error("No result files found. Please ensure your CSV files are in the 'results' directory.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("⚙️ Filters")

# Extract unique categories and difficulties from the original data
original_categories_set = set()
all_difficulties = set()
for model_df in all_results_data_original.values():
    if 'category' in model_df.columns:
        original_categories_set.update(model_df['category'].dropna().unique())
    if 'difficulty' in model_df.columns:
        all_difficulties.update(model_df['difficulty'].dropna().unique())

# Get grouped categories and the mapping
category_groups, sorted_group_names, original_cat_to_group_map = get_category_group_mapping_and_list(original_categories_set)
sorted_difficulties = sorted(list(all_difficulties))

# Determine default selected category groups based on top 10 original categories
all_original_categories_list = []
for model_df_original in all_results_data_original.values(): # Iterate over original data
    if 'category' in model_df_original.columns:
        all_original_categories_list.extend(model_df_original['category'].dropna().tolist())

default_selected_category_groups = sorted_group_names # Fallback to all groups

if all_original_categories_list:
    # Calculate frequencies of original categories
    s = pd.Series(all_original_categories_list)
    if not s.empty: # Ensure series is not empty before value_counts
        top_10_original_categories = s.value_counts().nlargest(10).index.tolist()
        
        target_groups_for_default = set()
        for cat_orig in top_10_original_categories:
            group = original_cat_to_group_map.get(cat_orig)
            if group:
                target_groups_for_default.add(group)
        
        if target_groups_for_default:
            default_selected_category_groups = sorted(list(target_groups_for_default))
    # If all_original_categories_list was empty or top_10 did not map, 
    # default_selected_category_groups remains sorted_group_names (all groups)

# Ensure "Model Building/Training (General)" is included by default if it's a valid group
model_building_group_name = "Model Building/Training (General)"
if model_building_group_name in sorted_group_names: # Check if it's a valid group name
    # Convert to set for efficient addition and to avoid duplicates, then back to sorted list
    current_defaults_set = set(default_selected_category_groups)
    current_defaults_set.add(model_building_group_name)
    default_selected_category_groups = sorted(list(current_defaults_set))

# Category Group selection
st.sidebar.subheader("🏷️ Task Category Groups")
selected_category_groups = st.sidebar.multiselect(
    "Select Task Category Groups:",
    options=sorted_group_names,
    default=default_selected_category_groups # Use the new dynamic default
)

# Determine which original categories are part of the selected groups
active_original_categories = []
for group_name in selected_category_groups:
    if group_name in category_groups:
        active_original_categories.extend(category_groups[group_name])
# Remove duplicates if any original category was somehow in multiple selected groups (shouldn't happen with current mapping)
active_original_categories = sorted(list(set(active_original_categories)))

# Difficulty selection
st.sidebar.subheader("📈 Task Difficulty")
selected_difficulties = st.sidebar.multiselect(
    "Select Task Difficulties:",
    options=sorted_difficulties,
    default=sorted_difficulties
)

# Filter data based on selected *original* categories (derived from groups) and difficulties
filtered_results_data = {}
if not all_results_data_original:
    st.error("Original data not loaded.")
    st.stop()

for model_name, df in all_results_data_original.items():
    df_filtered = df.copy()
    # Filter by original categories that fall under the selected groups
    if selected_category_groups and 'category' in df_filtered.columns:
        # Only keep rows where the category is in the list of active original categories
        df_filtered = df_filtered[df_filtered['category'].isin(active_original_categories)]
    
    if selected_difficulties and 'difficulty' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['difficulty'].isin(selected_difficulties)]
    
    if not df_filtered.empty:
        filtered_results_data[model_name] = df_filtered

if not filtered_results_data:
    st.warning("No data matches the selected category/difficulty filters. Please adjust your filter selection.")
    # We don't stop here, allow model selection to proceed with an empty summary_df if needed
    # which will then show appropriate messages in the main panel.
    summary_df = pd.DataFrame() # Ensure summary_df is an empty DataFrame
else:
    # Calculate summary statistics for all models based on filtered data
    summary_stats_list = []
    for model_name, df_filtered in filtered_results_data.items():
        stats = calculate_model_stats(model_name, df_filtered, PRICING_CONFIG)
        if stats:
            summary_stats_list.append(stats)

    if not summary_stats_list:
        # This case means filtered_results_data was not empty, but no stats could be calculated
        # (e.g., all remaining tasks had no valid scores)
        st.warning("Could not calculate statistics for any model with the current filters. All tasks might have invalid scores after filtering.")
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame(summary_stats_list)

# Model selection
st.sidebar.markdown("---") # Separator
st.sidebar.subheader("🤖 Model Selection")

if not summary_df.empty:
    # --- New Filters for Model Performance ---
    st.sidebar.markdown("#### Filter by Performance Metrics")
    avg_score_range = st.sidebar.slider(
        "Average Score Range (0-10):",
        min_value=0.0, max_value=10.0,
        value=(0.0, 10.0), step=0.1,
        help="Show models with an average score within this range."
    )
    success_rate_range = st.sidebar.slider(
        "Success Rate Range (%):",
        min_value=0.0, max_value=100.0,
        value=(0.0, 100.0), step=1.0,
        help="Show models with a success rate (score >= 8) within this range."
    )

    # Determine dynamic ranges for cost and score_per_dollar filters
    min_total_cost = 0.0
    max_total_cost = 10.0 # Default max if no data
    if 'total_cost_benchmark' in summary_df.columns and not summary_df['total_cost_benchmark'].dropna().empty:
        min_total_cost = summary_df['total_cost_benchmark'].min()
        max_total_cost = summary_df['total_cost_benchmark'].max()
        if pd.isna(min_total_cost): min_total_cost = 0.0
        if pd.isna(max_total_cost) or max_total_cost == min_total_cost: max_total_cost = min_total_cost + 10.0


    min_score_per_dollar = 0.0
    max_score_per_dollar = 100.0 # Default max if no data
    if 'score_per_dollar_benchmark' in summary_df.columns and not summary_df['score_per_dollar_benchmark'].dropna().empty:
        min_score_per_dollar = summary_df['score_per_dollar_benchmark'].min()
        max_score_per_dollar = summary_df['score_per_dollar_benchmark'].max()
        if pd.isna(min_score_per_dollar): min_score_per_dollar = 0.0
        if pd.isna(max_score_per_dollar) or max_score_per_dollar == min_score_per_dollar: max_score_per_dollar = min_score_per_dollar + 100.0
        # Ensure max is greater than min for slider
        if max_score_per_dollar <= min_score_per_dollar:
            max_score_per_dollar = min_score_per_dollar + 1.0 # Add a small delta if min and max are too close or equal

    total_cost_range = st.sidebar.slider(
        "Total Cost Range ($):",
        min_value=float(min_total_cost), max_value=float(max_total_cost),
        value=(float(min_total_cost), float(max_total_cost)), step=0.001 if max_total_cost < 1 else 0.1, # Finer step for smaller costs
        help="Show models with a total estimated cost for 180 tasks within this range."
    )
    score_per_dollar_range = st.sidebar.slider(
        "Score per Dollar Range:",
        min_value=float(min_score_per_dollar), max_value=float(max_score_per_dollar),
        value=(float(min_score_per_dollar), float(max_score_per_dollar)), step=0.1 if max_score_per_dollar < 10 else 1.0,
        help="Show models with a score per dollar (benchmark) within this range."
    )

    # Apply performance filters to the summary_df to get available models
    models_passing_performance_filters = summary_df[
        (summary_df['avg_score'] >= avg_score_range[0]) & (summary_df['avg_score'] <= avg_score_range[1]) &
        (summary_df['success_rate_percent'] >= success_rate_range[0]) & (summary_df['success_rate_percent'] <= success_rate_range[1]) &
        (summary_df['total_cost_benchmark'] >= total_cost_range[0]) & (summary_df['total_cost_benchmark'] <= total_cost_range[1]) &
        (summary_df['score_per_dollar_benchmark'] >= score_per_dollar_range[0]) & (summary_df['score_per_dollar_benchmark'] <= score_per_dollar_range[1])
    ]

    if not models_passing_performance_filters.empty:
        all_model_names = sorted(models_passing_performance_filters['model'].unique())
        st.sidebar.markdown("--- ") # Separator before model list
        st.sidebar.markdown("#### Select Models to Display")
        select_all_models = st.sidebar.checkbox("Select/Deselect All Filtered Models", value=True)

        if select_all_models:
            default_selected_models = all_model_names
        else:
            default_selected_models = []

        selected_models = st.sidebar.multiselect(
            "Select Models from Filtered List:",
            options=all_model_names,
            default=default_selected_models,
            help="These models meet the category, difficulty, and performance filters set above."
        )
    else:
        st.sidebar.info("No models meet the current performance filter criteria (average score/success rate). Adjust the sliders above.")
        all_model_names = [] # Ensure this is empty
        selected_models = [] # Ensure this is empty

elif summary_df.empty: # summary_df was empty to begin with (due to category/difficulty filters)
    st.sidebar.info("No models available to select based on current filters.")
    all_model_names = []
    selected_models = []


if not selected_models and not summary_df.empty: # summary_df might be empty if no data after cat/diff filters
    st.warning("Please select at least one model to display results.")
    st.stop()
elif not selected_models and summary_df.empty:
    # This means no models were available to select in the first place
    pass # Allow to proceed, main panel will show "No data..."

# Filter summary DataFrame based on model selection
if not summary_df.empty and selected_models:
    filtered_summary_df = summary_df[summary_df['model'].isin(selected_models)]
else:
    filtered_summary_df = pd.DataFrame() # Empty df if no models selected or summary_df was initially empty

# --- Main Panel ---
st.title("📊 Datascience LLM Benchmark Analysis")

st.header("🚀 Model Performance Overview")
st.caption("High-level statistics and comparisons of selected LLM models based on the applied filters.")

# Display filtered summary statistics as a table
st.subheader("📈 Summary Statistics")
st.caption("Key performance indicators for each selected model, reflecting the active category and difficulty filters. All costs are estimated for processing 180 benchmark tasks.")

# New metric: Number of tasks after category/difficulty filters
if filtered_results_data:
    # All DFs in filtered_results_data have been filtered by category/difficulty
    # and should have the same number of rows if they originated from the same task set.
    # This length represents the number of unique tasks that passed the filters.
    num_tasks_after_cat_diff_filters = len(next(iter(filtered_results_data.values())))
    st.metric(label="Active Benchmark Tasks (After Filters)", value=f"{num_tasks_after_cat_diff_filters} / 180")
else:
    # No data for any model passed the category/difficulty filters
    st.metric(label="Active Benchmark Tasks (After Filters)", value="0 / 180")

if not filtered_summary_df.empty:
    display_cols = ['model', 'avg_score', 'total_cost_benchmark', 'score_per_dollar_benchmark', 'success_rate_percent', 'fail_incomplete_rate_percent'] # total_tasks_processed already removed
    renamed_cols = {
        'model': 'Model',
        'avg_score': 'Average Score (0-10)',
        'total_cost_benchmark': 'Total Cost for 180 Tasks ($)',
        'score_per_dollar_benchmark': 'Score per Dollar (Benchmark)',
        # 'total_tasks_processed': 'Tasks Processed (after filters)', # Removed
        'success_rate_percent': 'Success Rate (%)',
        'fail_incomplete_rate_percent': 'Fail+Incomplete Rate (%)'
    }
    st.dataframe(filtered_summary_df[display_cols].rename(columns=renamed_cols).style.format({
        'Average Score (0-10)': '{:.2f}',
        'Total Cost for 180 Tasks ($)': '{:.3f}',
        'Score per Dollar (Benchmark)': '{:.2f}',
        'Success Rate (%)': '{:.1f}',
        'Fail+Incomplete Rate (%)': '{:.1f}'
    }))
else:
    st.info("No summary data to display for the selected models and filters.")


# --- Visualizations ---
st.header("🎨 Performance Visualizations")
st.caption("Visual comparisons of model performance across various metrics. All charts dynamically update based on the sidebar filters.")

if not filtered_summary_df.empty:
    # 1. Average Score Comparison (Bar Chart) - Full Width
    st.subheader("🏆 Average Score Comparison")
    st.caption("Compares the average evaluation score (0-10) of the selected models. Higher scores indicate better overall performance on the filtered tasks.")
    avg_score_sorted = filtered_summary_df.sort_values('avg_score', ascending=False)
    
    fig_avg_score, ax_avg_score = plt.subplots(figsize=(10, 6)) 
    bars = sns.barplot(x='avg_score', y='model', data=avg_score_sorted, ax=ax_avg_score, palette='viridis')
    ax_avg_score.set_xlabel("Average Score (0-10)")
    ax_avg_score.set_ylabel("Model")
    ax_avg_score.set_title("Average Score per Model")
    for bar in bars.patches:
        ax_avg_score.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, 
                         f'{bar.get_width():.2f}', 
                         va='center', ha='left')
    plt.tight_layout()
    st.pyplot(fig_avg_score)

    # Create columns for side-by-side heatmaps
    col1, col2 = st.columns(2)

    with col1:
        # 3. Scores by Difficulty Heatmap
        st.subheader("🌡️ Avg Scores by Difficulty")
        st.caption("Heatmap showing the average score of each model across different task difficulty levels. Helps identify model strengths/weaknesses at varying complexities.")
        
        difficulty_heatmap_data_list = []
        if filtered_results_data and selected_models:
            for model_name in selected_models:
                if model_name in filtered_results_data:
                    df_model = filtered_results_data[model_name]
                    if 'difficulty' in df_model.columns and 'evaluation_score' in df_model.columns:
                        df_model_copy = df_model.copy()
                        df_model_copy['evaluation_score'] = pd.to_numeric(df_model_copy['evaluation_score'], errors='coerce')
                        df_model_cleaned = df_model_copy.dropna(subset=['evaluation_score', 'difficulty'])

                        if not df_model_cleaned.empty:
                            difficulty_scores = df_model_cleaned.groupby('difficulty')['evaluation_score'].mean().reset_index()
                            difficulty_scores['model'] = model_name
                            difficulty_scores.rename(columns={'evaluation_score': 'avg_score_in_difficulty'}, inplace=True)
                            difficulty_heatmap_data_list.append(difficulty_scores)
        
        if difficulty_heatmap_data_list:
            all_difficulty_scores_df = pd.concat(difficulty_heatmap_data_list)
            
            if not all_difficulty_scores_df.empty:
                try:
                    heatmap_pivot = all_difficulty_scores_df.pivot_table(
                        index='model', 
                        columns='difficulty', 
                        values='avg_score_in_difficulty'
                    )
                    heatmap_pivot.fillna(0, inplace=True) 

                    if not heatmap_pivot.empty:
                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(heatmap_pivot.columns) * 1.5), max(6, len(heatmap_pivot.index) * 0.4)))
                        sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_heatmap, linewidths=.5, cbar=True)
                        ax_heatmap.set_title("Avg Scores by Task Difficulty")
                        ax_heatmap.set_xlabel("Task Difficulty")
                        ax_heatmap.set_ylabel("Model")
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig_heatmap)
                    else:
                        st.info("Not enough data for difficulty heatmap (pivot empty).")
                except Exception as e:
                    st.error(f"Error generating difficulty heatmap: {e}")
                    st.info("Could not generate difficulty heatmap.")
            else:
                st.info("No difficulty score data for heatmap.")
        else:
            st.info("No data for difficulty heatmap.")

    with col2:
        # 4. Scores by Category Group Heatmap
        st.subheader("📚 Avg Scores by Category Group")
        st.caption("Heatmap displaying average scores for each model within defined task category groups. Useful for understanding performance in specific domains.")
        category_heatmap_data_list = []
        if filtered_results_data and selected_models:
            for model_name in selected_models:
                if model_name in filtered_results_data:
                    df_model = filtered_results_data[model_name].copy()
                    if 'category' in df_model.columns and 'evaluation_score' in df_model.columns:
                        # Map original categories to their groups for this model's data
                        df_model['category_group'] = df_model['category'].map(original_cat_to_group_map)
                        df_model_copy = df_model.copy()
                        df_model_copy['evaluation_score'] = pd.to_numeric(df_model_copy['evaluation_score'], errors='coerce')
                        df_model_cleaned = df_model_copy.dropna(subset=['evaluation_score', 'category_group'])

                        if not df_model_cleaned.empty:
                            category_group_scores = df_model_cleaned.groupby('category_group')['evaluation_score'].mean().reset_index()
                            category_group_scores['model'] = model_name
                            category_group_scores.rename(columns={'evaluation_score': 'avg_score_in_group'}, inplace=True)
                            category_heatmap_data_list.append(category_group_scores)
        
        if category_heatmap_data_list:
            all_category_group_scores_df = pd.concat(category_heatmap_data_list)
            if not all_category_group_scores_df.empty:
                try:
                    cat_heatmap_pivot = all_category_group_scores_df.pivot_table(
                        index='model',
                        columns='category_group',
                        values='avg_score_in_group'
                    )
                    cat_heatmap_pivot.fillna(0, inplace=True)

                    if not cat_heatmap_pivot.empty:
                        fig_cat_heatmap, ax_cat_heatmap = plt.subplots(figsize=(max(10, len(cat_heatmap_pivot.columns) * 1.2), max(6, len(cat_heatmap_pivot.index) * 0.5)))
                        sns.heatmap(cat_heatmap_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_cat_heatmap, linewidths=.5, cbar=True)
                        ax_cat_heatmap.set_title("Average Scores by Task Category Group")
                        ax_cat_heatmap.set_xlabel("Task Category Group")
                        ax_cat_heatmap.set_ylabel("Model")
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig_cat_heatmap)
                    else:
                        st.info("Not enough data for category group heatmap (pivot empty).")
                except Exception as e:
                    st.error(f"Error generating category group heatmap: {e}")
                    st.info("Could not generate category group heatmap.")
            else:
                st.info("No category group score data for heatmap.")
        else:
            st.info("No data for category group heatmap.")

    # NEW: Average Score per Category Group (Aggregated for selected models)
    st.subheader("🌐 Avg Score by Category Group (Aggregated)")
    st.caption("Shows the average performance of all selected models combined, for each task category group. Helps identify which task types are generally easier or harder for the chosen set of models.")
    if filtered_results_data and selected_models:
        aggregated_category_scores_list = []
        for model_name in selected_models:
            if model_name in filtered_results_data:
                df_model = filtered_results_data[model_name].copy()
                if 'category' in df_model.columns and 'evaluation_score' in df_model.columns:
                    df_model['category_group'] = df_model['category'].map(original_cat_to_group_map)
                    df_model['evaluation_score'] = pd.to_numeric(df_model['evaluation_score'], errors='coerce')
                    df_model_cleaned = df_model.dropna(subset=['evaluation_score', 'category_group'])
                    if not df_model_cleaned.empty:
                        # We need to group by category_group and then average these scores across models later
                        aggregated_category_scores_list.append(df_model_cleaned[['category_group', 'evaluation_score']])
        
        if aggregated_category_scores_list:
            all_models_category_data = pd.concat(aggregated_category_scores_list)
            if not all_models_category_data.empty:
                avg_scores_by_cat_group = all_models_category_data.groupby('category_group')['evaluation_score'].mean().sort_values(ascending=False)
                
                if not avg_scores_by_cat_group.empty:
                    fig_agg_cat_score, ax_agg_cat_score = plt.subplots(figsize=(12, max(6, len(avg_scores_by_cat_group) * 0.5)))
                    sns.barplot(x=avg_scores_by_cat_group.values, y=avg_scores_by_cat_group.index, ax=ax_agg_cat_score, palette="crest")
                    ax_agg_cat_score.set_xlabel("Average Score (0-10)")
                    ax_agg_cat_score.set_ylabel("Task Category Group")
                    ax_agg_cat_score.set_title("Average Score by Category Group (All Selected Models)")
                    for i, v in enumerate(avg_scores_by_cat_group.values):
                        ax_agg_cat_score.text(v + 0.02, i, f'{v:.2f}', va='center', ha='left')
                    plt.tight_layout()
                    st.pyplot(fig_agg_cat_score)
                else:
                    st.info("No aggregated category scores to display.")
            else:
                st.info("No data to aggregate for category group scores.")
        else:
            st.info("No data from selected models for aggregated category group scores.")
    else:
        st.info("Select models and ensure data is available for aggregated category scores.")

    # 2. Cost vs Performance (Scatter Plot) - Now displayed below the two columns, full width
    st.subheader("💰 Cost vs. Average Performance")
    st.caption("Scatter plot illustrating the relationship between the total estimated cost (log scale) and average performance. Bubble size represents 'Score per Dollar'. Ideal models are in the top-left (high score, low cost).")
    priced_models = filtered_summary_df[filtered_summary_df['total_cost_benchmark'] > 0]
    
    if not priced_models.empty:
        fig_cost_perf, ax_cost_perf = plt.subplots(figsize=(12, 7)) # Increased figsize
        scatter = sns.scatterplot(
            x='total_cost_benchmark', 
            y='avg_score', 
            size='score_per_dollar_benchmark', 
            hue='model', 
            data=priced_models, 
            ax=ax_cost_perf, 
            sizes=(50, 500), 
            alpha=0.7,
            legend=False 
        )
        ax_cost_perf.set_xlabel("Total Cost for 180 Tasks ($) - Log Scale")
        ax_cost_perf.set_ylabel("Average Score (0-10)")
        ax_cost_perf.set_title("Cost vs. Performance (Bubble size = Score/Dollar)")
        ax_cost_perf.set_xscale('log') # Apply log scale to x-axis
        ax_cost_perf.set_ylim(0, 10) 
        
        for i, row in priced_models.iterrows():
            ax_cost_perf.annotate(row['model'], (row['total_cost_benchmark'], row['avg_score']), 
                                 xytext=(7,1), textcoords='offset points', fontsize=9) # Adjusted xytext
        
        plt.tight_layout()
        st.pyplot(fig_cost_perf)
    else:
        st.info("No models with pricing data selected or available for the Cost vs. Performance chart with current filters.")

    # NEW: Success Rate vs Total Cost (Scatter Plot)
    st.subheader("🎯 Success Rate vs. Total Cost")
    st.caption("Visualizes model success rates (tasks scoring >= 8) against their total estimated cost (log scale). Helps assess cost-effectiveness in achieving successful outcomes.")
    # We need 'success_rate_percent' and 'total_cost_benchmark' from filtered_summary_df
    # Ensure models have pricing data for this chart as well
    priced_success_models = filtered_summary_df[(filtered_summary_df['total_cost_benchmark'] > 0) & (filtered_summary_df['success_rate_percent'].notna())]

    if not priced_success_models.empty:
        fig_sr_cost, ax_sr_cost = plt.subplots(figsize=(12, 7))
        sns.scatterplot(
            x='total_cost_benchmark',
            y='success_rate_percent',
            hue='model',
            data=priced_success_models,
            ax=ax_sr_cost,
            s=100, # Fixed size for this plot, or could be dynamic based on another metric
            alpha=0.7,
            legend=False
        )
        ax_sr_cost.set_xlabel("Total Cost for 180 Tasks ($) - Log Scale")
        ax_sr_cost.set_ylabel("Success Rate (%)")
        ax_sr_cost.set_title("Success Rate vs. Total Cost")
        ax_sr_cost.set_xscale('log')
        ax_sr_cost.set_ylim(0, 100) # Success rate is 0-100

        for i, row in priced_success_models.iterrows():
            ax_sr_cost.annotate(row['model'], (row['total_cost_benchmark'], row['success_rate_percent']), 
                                 xytext=(7,1), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_sr_cost)
    else:
        st.info("No models with pricing and success rate data available for the Success Rate vs. Total Cost chart with current filters.")


    # --- Task Outcome Analysis ---
    st.subheader("📊 Task Outcome Analysis")
    st.caption("Breakdown of task outcomes (Successful, Incomplete, Failed) for each model, shown in both absolute counts and percentages.")

    outcome_data_list = []
    if filtered_results_data and selected_models:
        for model_name in selected_models:
            if model_name in filtered_results_data:
                df_model = filtered_results_data[model_name].copy() # Use .copy() to avoid SettingWithCopyWarning
                if 'evaluation_score' in df_model.columns:
                    df_model['evaluation_score'] = pd.to_numeric(df_model['evaluation_score'], errors='coerce')
                    df_model.dropna(subset=['evaluation_score'], inplace=True)

                    if not df_model.empty:
                        conditions = [
                            df_model['evaluation_score'] <= 5,
                            (df_model['evaluation_score'] >= 6) & (df_model['evaluation_score'] <= 7),
                            df_model['evaluation_score'] >= 8
                        ]
                        choices = ['Failed (<=5)', 'Incomplete (6-7)', 'Successful (>=8)']
                        df_model['outcome_status'] = pd.Series(pd.NA) # Initialize with NA
                        df_model['outcome_status'] = np.select(conditions, choices, default=pd.NA)
                        
                        # Filter out rows where outcome_status might be NA if no condition was met (should not happen with these conditions)
                        df_model.dropna(subset=['outcome_status'], inplace=True)

                        if not df_model.empty:
                            outcome_counts = df_model['outcome_status'].value_counts().reset_index()
                            outcome_counts.columns = ['outcome_status', 'count']
                            outcome_counts['model'] = model_name
                            outcome_data_list.append(outcome_counts)
    
    if outcome_data_list:
        all_outcome_counts_df = pd.concat(outcome_data_list)

        if not all_outcome_counts_df.empty:
            # 1. Absolute Counts Bar Chart
            st.markdown("##### Task Outcome Counts")
            fig_outcome_counts, ax_outcome_counts = plt.subplots(figsize=(12, 7))
            sns.barplot(x='model', y='count', hue='outcome_status', data=all_outcome_counts_df, 
                        ax=ax_outcome_counts, palette={'Successful (>=8)': 'green', 'Incomplete (6-7)': 'orange', 'Failed (<=5)': 'red'},
                        hue_order=['Successful (>=8)', 'Incomplete (6-7)', 'Failed (<=5)'])
            ax_outcome_counts.set_title("Task Outcome Counts per Model")
            ax_outcome_counts.set_xlabel("Model")
            ax_outcome_counts.set_ylabel("Number of Tasks")
            ax_outcome_counts.legend(title="Outcome Status")
            ax_outcome_counts.yaxis.grid(True, linestyle='--', alpha=0.7) # Add horizontal grid lines
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_outcome_counts)

            # 2. Percentage Stacked Bar Chart
            st.markdown("##### Task Outcome Percentages")
            st.caption("Stacked bar chart showing the percentage distribution of task outcomes for each model. Allows quick assessment of model performance consistency.")
            model_totals = all_outcome_counts_df.groupby('model')['count'].sum()
            
            # Calculate percentages
            percentage_df = all_outcome_counts_df.copy()
            percentage_df['percentage'] = percentage_df.apply(
                lambda row: (row['count'] / model_totals[row['model']]) * 100 if model_totals[row['model']] > 0 else 0, 
                axis=1
            )
            
            # Pivot for stacked bar chart
            status_order = ['Successful (>=8)', 'Incomplete (6-7)', 'Failed (<=5)']
            percentage_pivot = percentage_df.pivot_table(
                index='model', 
                columns='outcome_status', 
                values='percentage',
                fill_value=0
            )[status_order] # Ensure correct order for stacking

            if not percentage_pivot.empty:
                fig_outcome_perc, ax_outcome_perc = plt.subplots(figsize=(12, 7))
                percentage_pivot.plot(kind='bar', stacked=True, 
                                      ax=ax_outcome_perc, 
                                      color={'Successful (>=8)': 'green', 'Incomplete (6-7)': 'orange', 'Failed (<=5)': 'red'})
                ax_outcome_perc.set_title("Task Outcome Percentages per Model")
                ax_outcome_perc.set_xlabel("Model")
                ax_outcome_perc.set_ylabel("Percentage of Tasks (%)")
                ax_outcome_perc.legend(title="Outcome Status", bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_outcome_perc.set_ylim(0, 100) # Ensure y-axis is 0-100 for percentages
                
                # Add annotations for percentages
                for i, model_name in enumerate(percentage_pivot.index):
                    cumulative_height = 0
                    for status in status_order:
                        value = percentage_pivot.loc[model_name, status]
                        if value > 0: # Only annotate if value is not zero
                            # Place text in the middle of the bar segment
                            ax_outcome_perc.text(i, cumulative_height + value / 2, f"{value:.1f}%", 
                                                 ha='center', va='center', color='black', fontsize=6, fontweight='bold') # Changed fontsize to 7
                        cumulative_height += value

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
                st.pyplot(fig_outcome_perc)
            else:
                st.info("Not enough data to generate the outcome percentage chart.")
        else:
            st.info("No task outcome data available for the selected models and filters.")
    else:
        st.info("No data available for task outcome analysis based on current selections.")

    # --- Success and Unsuccessful Rate Graphs ---
    st.subheader("⚖️ Success vs. Unsuccessful Task Rates")
    st.caption("Side-by-side comparison of success rates (score >= 8) and combined fail/incomplete rates (score <= 7) for each model.")

    if outcome_data_list: # Re-use outcome_data_list processed earlier
        all_outcome_counts_df = pd.concat(outcome_data_list)
        if not all_outcome_counts_df.empty:
            # Calculate total tasks per model again or pass it from previous calculation
            model_totals = all_outcome_counts_df.groupby('model')['count'].sum()

            success_data = []
            unsuccessful_data = []

            for model_name in model_totals.index:
                model_df = all_outcome_counts_df[all_outcome_counts_df['model'] == model_name]
                total_tasks_for_model = model_totals[model_name]

                successful_count = model_df[model_df['outcome_status'] == 'Successful (>=8)']['count'].sum()
                incomplete_count = model_df[model_df['outcome_status'] == 'Incomplete (6-7)']['count'].sum()
                failed_count = model_df[model_df['outcome_status'] == 'Failed (<=5)']['count'].sum()
                
                unsuccessful_count = incomplete_count + failed_count

                success_rate = (successful_count / total_tasks_for_model) * 100 if total_tasks_for_model > 0 else 0
                unsuccessful_rate = (unsuccessful_count / total_tasks_for_model) * 100 if total_tasks_for_model > 0 else 0
                
                success_data.append({'model': model_name, 'rate': success_rate, 'type': 'Successful'})
                unsuccessful_data.append({'model': model_name, 'rate': unsuccessful_rate, 'type': 'Unsuccessful'})

            success_df = pd.DataFrame(success_data)
            unsuccessful_df = pd.DataFrame(unsuccessful_data)

            col1, col2 = st.columns(2)

            with col1:
                if not success_df.empty:
                    st.markdown("##### Success Rate (Score >= 8)")
                    fig_success, ax_success = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='rate', y='model', data=success_df.sort_values('rate', ascending=False), ax=ax_success, color='green')
                    ax_success.set_xlabel("Success Rate (%)")
                    ax_success.set_ylabel("Model")
                    ax_success.set_title("Task Success Rate")
                    ax_success.set_xlim(0, 100)
                    for bar in ax_success.patches:
                        ax_success.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, 
                                         f'{bar.get_width():.1f}%', va='center', ha='left')
                    plt.tight_layout()
                    st.pyplot(fig_success)
                else:
                    st.info("No data for success rate graph.")
            
            with col2:
                if not unsuccessful_df.empty:
                    st.markdown("##### Fail + Incomplete (Score <= 7)")
                    fig_unsuccessful, ax_unsuccessful = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='rate', y='model', data=unsuccessful_df.sort_values('rate', ascending=False), ax=ax_unsuccessful, color='red')
                    ax_unsuccessful.set_xlabel("Fail + Incomplete Rate (%)")
                    ax_unsuccessful.set_ylabel("") # Remove y-label to avoid repetition
                    ax_unsuccessful.set_title("Task Unsuccessful Rate")
                    ax_unsuccessful.set_xlim(0, 100)
                    for bar in ax_unsuccessful.patches:
                        ax_unsuccessful.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, 
                                             f'{bar.get_width():.1f}%', va='center', ha='left')
                    plt.tight_layout()
                    st.pyplot(fig_unsuccessful)
                else:
                    st.info("No data for unsuccessful rate graph.")
        else:
            st.info("No task outcome data to calculate success/unsuccessful rates.")
    else:
        st.info("No data available for success/unsuccessful rate analysis.")

    # --- Hall of Fame ---
    st.header("🌟 Hall of Fame")
    st.caption("Highlighting top-performing models based on different criteria, reflecting current filters.")

    if not filtered_summary_df.empty:
        # Best Pure Performance (Highest Average Score)
        st.subheader("🥇 Top Performers (by Average Score)")
        st.caption("Models with the highest overall average scores on the filtered tasks.")
        top_performers = filtered_summary_df.sort_values(by='avg_score', ascending=False).head(5) # Top 5
        if not top_performers.empty:
            st.dataframe(
                top_performers[['model', 'avg_score', 'total_cost_benchmark', 'success_rate_percent']]
                .rename(columns=renamed_cols)
                .style.format({
                    'Average Score (0-10)': '{:.2f}',
                    'Total Cost for 180 Tasks ($)': '{:.3f}',
                    'Success Rate (%)': '{:.1f}'
                })
            )
        else:
            st.info("Not enough data to display top performers.")

        # Best Price-to-Performance (Highest Score per Dollar)
        # Exclude models with zero or negligible cost to avoid division by zero or extremely high ratios
        st.subheader("💸 Best Value (by Score per Dollar)")
        st.caption("Models offering the best performance relative to their estimated cost. Calculated as 'Average Score / Total Cost'.")
        value_performers = filtered_summary_df[filtered_summary_df['total_cost_benchmark'] > 0.001] # Avoid division by zero or near-zero
        if not value_performers.empty:
            value_performers = value_performers.sort_values(by='score_per_dollar_benchmark', ascending=False).head(5) # Top 5
            if not value_performers.empty:
                st.dataframe(
                    value_performers[['model', 'score_per_dollar_benchmark', 'avg_score', 'total_cost_benchmark']]
                    .rename(columns=renamed_cols)
                    .style.format({
                        'Score per Dollar (Benchmark)': '{:.2f}',
                        'Average Score (0-10)': '{:.2f}',
                        'Total Cost for 180 Tasks ($)': '{:.3f}'
                    })
                )
            else:
                st.info("Not enough data to display best value performers (all models might have zero cost or no score/dollar calculated).")
        else:
            st.info("No models with significant cost to calculate score per dollar for best value.")
    else:
        st.info("No summary data available for Hall of Fame.")


    # Detailed view per model (optional, can be expanded)
    st.header("🔍 Detailed Model Data")
    st.caption("View the raw, filtered benchmark scores for a specific model selected from the dropdown below.")
    
    # Ensure selected_models for detail dropdown are from those that have data after all filters
    # These are the models present in filtered_results_data.keys()
    # And also selected in the model multiselect
    
    available_models_for_detail = [m for m in selected_models if m in filtered_results_data]

    if available_models_for_detail:
        selected_model_for_detail = st.selectbox(
            "Select a model to see its filtered raw scores:", 
            options=available_models_for_detail
        )
        if selected_model_for_detail and selected_model_for_detail in filtered_results_data:
            st.subheader(f"Filtered Raw Scores for {selected_model_for_detail}")
            st.dataframe(filtered_results_data[selected_model_for_detail])
        elif selected_model_for_detail: # Model was in dropdown but somehow not in filtered_results_data (should not happen)
             st.info(f"No detailed data available for {selected_model_for_detail} with current filters.")
    else:
        st.info("No models available for detailed view with current filters.")
else:
    st.info("No data available for the selected models and filters to generate charts.")
