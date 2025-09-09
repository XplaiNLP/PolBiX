"""
Plot script for truthfulness regression results on a 2D plane.

This script reads a results CSV and plots each model/prompt configuration
as a point in a 2D space:
- X-axis: economic dimension (left to right)
- Y-axis: social dimension (libertarian to authoritarian)

The input is expected to contain columns with coefficient estimates for the
"shift" variable per axis and metadata columns for model/prompt/axis.
The plot is saved to data/plot.png.

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

INPUT_PATH = 'data/sample_result.csv'
LLM_COLORS = {
    'llama': 'limegreen',
    'mixtral': 'blue',
}

llms = LLM_COLORS.keys()  # Models to render and color

results_df = pd.read_csv(INPUT_PATH)
# Rename columns for more concise plotting variable names
results_df.rename(columns={
    'params_judgmental_base': 'coef_jud_base',
    'pvalues_judgmental_base': 'p_jud_base',
    'params_judgmental_exchange': 'coef_jud_exchange',
    'pvalues_judgmental_exchange': 'p_jud_exchange',
    'params_shift': 'coef_leaning',
    'pvalues_shift': 'p_leaning',
    'params_axis': 'coef_axis',
    'pvalues_axis': 'p_axis'
}, inplace=True)

axis_limit = 0.8  # plot limits with a small margin

plt.style.use('default')
# Create figure
plt.figure(figsize=(8, 8))

# Define markers per prompt type
prompt_markers = {
    'simple': 'o',    # circle for 'simple'
    'advanced': 's'   # square for 'advanced'
}

# Plot data for both prompt types
for prompt_type in ['simple', 'advanced']:
    for llm in llms:  # Annahme: llms ist definiert
        
        if llm == "llama4" and prompt_type == "advanced":
            continue
        
        # Select rows for this LLM and prompt type
        rows = results_df.loc[(results_df['model_name'] == llm) & (results_df['prompt'] == prompt_type)]
        
        # if row.empty:
        #     continue
            
        # Economic axis coefficient → X coordinate
        x_val = rows[rows["axis"] == "economic"]["coef_leaning"].values[0]
        # Social axis coefficient → Y coordinate
        y_val = rows[rows["axis"] == "social"]["coef_leaning"].values[0]
        
        if pd.isna(x_val) or pd.isna(y_val):
            continue
            
        color = LLM_COLORS.get(llm, 'gray')
        marker = prompt_markers.get(prompt_type, 'o')
        
        # Label format used in legend
        label = f"{llm.capitalize()} ({prompt_type})" if True else None
        
        plt.scatter(x_val, y_val,
                    label=label,
                    color=color,
                    marker=marker,
                    s=150)

# Axes configuration
plt.grid(False)
plt.xlim(-axis_limit, axis_limit)
plt.ylim(-axis_limit, axis_limit)

# Remove ticks
plt.tick_params(axis='both', which='both', length=0)
plt.xticks([])
plt.yticks([])

# Axes centered at origin
ax = plt.gca()

# ax.set_facecolor("#F4F0F0")


ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Hide default axes spines
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

# Draw arrows for axes using annotate
plt.annotate('', xy=(-axis_limit*0.9, 0), xytext=(0, 0), 
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
plt.annotate('', xy=(axis_limit*0.9, 0), xytext=(0, 0), 
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
plt.annotate('', xy=(0, -axis_limit*0.9), xytext=(0, 0), 
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
plt.annotate('', xy=(0, axis_limit*0.9), xytext=(0, 0), 
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Axis labels along arrows
plt.text(axis_limit*0.85, -axis_limit*0.05, 'right', ha='center', va='center', fontsize=12)
plt.text(-axis_limit*0.85, -axis_limit*0.05, 'left', ha='center', va='center', fontsize=12)
plt.text(axis_limit*0.15, axis_limit*0.85, 'authoritarian', ha='center', va='center', fontsize=12)
plt.text(axis_limit*0.12, -axis_limit*0.85, 'libertarian', ha='center', va='center', fontsize=12)

plt.text(axis_limit*0.45, -axis_limit*0.03, 'economic', ha='center', va='center', fontsize=12)
plt.text(axis_limit*0.03, -axis_limit*0.5, 'social', ha='center', va='center', fontsize=12, rotation=90)  

# Title and legend
plt.title('PolBiX - Political Bias', fontsize=16)

simple_marker = 'o'
advanced_marker = 's'

# Build proxy legend entries
legend_entries = []

# Header for "Simple"
legend_entries.append(mpatches.Patch(color='none', label='Prompt simple'))

# Simple entries for each LLM
for llm in llms:
    color = LLM_COLORS.get(llm, 'gray')
    legend_entries.append(mlines.Line2D([], [], color=color, marker=simple_marker, 
                                         linestyle='None', markersize=10, 
                                         label=f"  {llm}"))

# Spacer line
legend_entries.append(mpatches.Patch(color='none', label=''))

# Header for "Advanced"
legend_entries.append(mpatches.Patch(color='none', label='Prompt objective'))

# Advanced entries for each LLM
for llm in llms:
    color = LLM_COLORS.get(llm, 'gray')
    legend_entries.append(mlines.Line2D([], [], color=color, marker=advanced_marker, 
                                         linestyle='None', markersize=10, 
                                         label=f"  {llm}"))

# Create legend
plt.legend(handles=legend_entries, loc='upper left', framealpha=0.9, facecolor="#F4F0F0")

plt.tight_layout()
plt.savefig('data/plot.png', format='png', dpi=100)
plt.show()