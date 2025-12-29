import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Competition.utils import *
import pandas as pd

def process_dataset_data(model_names_in_file, dataset_path):
    """
    Process the data of a single dataset and return the (real, virtual) SimMCV values for each model.
    """
    dataset_results = []
    
    for model_name in model_names_in_file:
        # Read JSON file
        data = read_json(f'{dataset_path}_{model_name}_analysis.json')
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(data)
        
        # Filter out English data and select the required columns
        english_data = []
        for item in data:
            if 'english' in item and 'percentage_simple' in item['english']:
                english_item = {
                    'model_name': item['model_name'],
                    'bias_type': item['bias_type'],
                    'test_mode': item['test_mode'],
                    'game_name': item['game_name'],
                    'percentage_simple': item['english']['percentage_simple']
                }
                english_data.append(english_item)
        
        df_english = pd.DataFrame(english_data)
        
        # Handle the case where test_mode is None (treat it as real)
        df_english['test_mode'] = df_english['test_mode'].apply(
            lambda x: 'real' if x is None else x
        )
        
        # Exclude data of age type (because age has no virtual data)
        df_filtered = df_english[df_english['bias_type'] != 'age']
        
        # Step 1: Group by bias_type and test_mode to calculate the average (handling different game_name cases)
        step1_avg = df_filtered.groupby(['bias_type', 'test_mode'])['percentage_simple'].mean().reset_index()
        
        # Step 2: Group by test_mode and calculate the average (average over the bias_type dimension)
        step2_avg = step1_avg.groupby('test_mode')['percentage_simple'].mean().reset_index()
        
        # Get the values of real and virtual
        real_value = step2_avg[step2_avg['test_mode'] == 'real']['percentage_simple'].values
        virtual_value = step2_avg[step2_avg['test_mode'] == 'virtual']['percentage_simple'].values
        
        # Ensure both values exist
        if len(real_value) > 0 and len(virtual_value) > 0:
            dataset_results.append([real_value[0], virtual_value[0]])
        else:
            # If a value is missing, fill it with NaN
            real_val = real_value[0] if len(real_value) > 0 else np.nan
            virtual_val = virtual_value[0] if len(virtual_value) > 0 else np.nan
            dataset_results.append([real_val, virtual_val])
    
    return dataset_results

def plot_r_n_v(model_names, data_1, data_2, data_3):
    # Data: (x, y) pairs for eight models across three datasets
    models = model_names
    group1 = np.array(data_1)
    group2 = np.array(data_2)
    group3 = np.array(data_3)
    
    # Configurations
    groups = [group1, group2, group3]
    # colors = ['tab:red', 'tab:green', 'tab:blue'] # dataset colors
    colors = ['teal', 'royalblue', 'lightcoral']

    # markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'v'] # unique marker per model
    markers = ['o', '^', 's', 'D', 'P', 'X', '*']# unique marker per model
    labels_datasets = ['Tr', 'Coo', 'Com']
    marker_size = 150                            # marker size
    line_width = 3.5                               # thicker regression lines

    # Plot
    plt.figure(figsize=(10, 9))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Scatter points: loop over models to keep marker consistent across datasets
    for idx_model, marker in enumerate(markers):
        for color, g in zip(colors, groups):
            x, y = g[idx_model]
            ax.scatter(x,
                       y,
                       color=color,
                       marker=marker,
                       s=marker_size,
                       edgecolors='k',
                       linewidths=1.5,
                       alpha=0.7
            )

    # Linear fits (one per dataset)
    for g, color, label in zip(groups, colors, labels_datasets):
        x_vals, y_vals = g[:, 0], g[:, 1]
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_fit,
                slope * x_fit + intercept,
                color=color,
                linestyle='--',
                linewidth=line_width,
                label=label
        )

    # Aesthetics
    ax.set_xlabel('FairMCV Score (Realistic Genre Data)(%)', fontsize=24)
    ax.set_ylabel('FairMCV Score (Virtual Genre Data)(%)', fontsize=24)

    # Set the axis range to 50-100
    ax.set_xlim(50, 100)
    ax.set_ylim(50, 100)

    # Removed title as requested
    ax.grid(True, linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=16)  # tick fontsize

    # Legends
    # --- Tasks legend (datasets) ---
    legend_tasks = ax.legend(title='Tasks', fontsize=22, title_fontsize=22,
                         loc='upper left', bbox_to_anchor=(0.01, 0.52))
    ax.add_artist(legend_tasks)

    # --- Model legend (markers) ---
    model_handles = [Line2D([], [], marker=markers[i], linestyle='None', color='k',
                        markeredgecolor='k', markersize=np.sqrt(marker_size), label=models[i])
                 for i in range(len(models))]
    legend_models = ax.legend(handles=model_handles, title='Models', fontsize=22, title_fontsize=22,
                          loc='upper left', bbox_to_anchor=(0.01, 1))

    # Layout & save
    plt.tight_layout()
    plt.savefig('./img/plot_r_n_v_v3.pdf', dpi=500)
    plt.savefig('./img/plot_r_n_v_v3.png', dpi=500)
    plt.show()

if __name__ == '__main__':
    # Data: (x, y) pairs for eight models across three datasets
    models = [
        'GPT-4.1', 'Grok-4', 'Grok-4-fast', 'DeepSeek-V3', 
        'Qwen2.5-72B', 'LLaMA-3.1 70B', 'LLaMA-3.1 8B'
    ]
    model_names_in_file = [
        "gpt-4.1", "grok-4-0709", "grok-4-fast-non-reasoning", "deepseek-chat", 
        "qwen2.5-72b-instruct", "Meta-Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct"
    ]
    
    # Process three datasets
    data_1 = process_dataset_data(model_names_in_file, './Trade/record/Tr')
    data_2 = process_dataset_data(model_names_in_file, './Cooperation/record/Coo') 
    data_3 = process_dataset_data(model_names_in_file, './Competition/record/Com')
    
    # Print processed data for verification
    print("Tr data:")
    for i, row in enumerate(data_1):
        print(f"  {models[i]}: {row}")
    
    print("\nCoo data:")
    for i, row in enumerate(data_2):
        print(f"  {models[i]}: {row}")
    
    print("\nCom data:")
    for i, row in enumerate(data_3):
        print(f"  {models[i]}: {row}")
    
    # Call the drawing function
    plot_r_n_v(models, data_1, data_2, data_3)