import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def extract_data():
    """
    Extract data from LaTeX tables
    """
    data = {
        'Model': ['GPT-4.1', 'Grok-4', 'Grok-4-fast', 
                  'DeepSeek-V3.2', 'Qwen2.5-72B', 'LLaMA-3.3 70B', 'LLaMA-3.1 8B'],
        
        # Career:
        'Career_Tr': [76.9, 78.8, 74.8, 80.7, 90.8, 91.4, 94.6],
        'Career_Coo': [84.1, 84.2, 82.0, 81.3, 84.0, 80.5, 84.5],
        'Career_Com': [76.7, 80.1, 75.8, 67.0, 73.1, 74.0, 77.9],
        
        # Race:
        'Race_Tr': [78.2, 72.9, 71.7, 80.9, 97.7, 94.0, 93.8],
        'Race_Coo': [95.2, 95.0, 93.9, 92.4, 92.9, 94.6, 91.1],
        'Race_Com': [66.6, 69.9, 63.7, 66.6, 72.9, 72.6, 75.2],
        
        # Age:
        'Age_Tr': [77.9, 83.0, 81.5, 82.1, 94.9, 92.7, 92.2],
        'Age_Coo': [83.0, 91.5, 89.4, 80.4, 83.6, 83.1, 87.9],
        'Age_Com': [64.2, 75.9, 67.0, 68.4, 68.9, 67.9, 78.5],
        
        # Nationality:
        'Nationality_Tr': [82.0, 69.5, 78.2, 80.9, 94.0, 96.2, 92.9],
        'Nationality_Coo': [95.7, 88.1, 83.3, 91.2, 94.0, 97.6, 88.4],
        'Nationality_Com': [66.0, 68.7, 61.6, 63.1, 77.7, 73.7, 74.0]
    }
    return pd.DataFrame(data)

def calculate_and_print_averages():
    """
    Calculate the mean for each model and print the results
    """
    # Extract data
    df = extract_data()
    
    # Calculate the mean for each model (excluding the Model column)
    averages = df.drop('Model', axis=1).mean(axis=1).round(1).tolist()
    
    # Output a list of floating-point numbers
    print("List of average scores for each model:")
    print(averages)
    print()
    
    # Output the average score for each model
    print("Average scores of each model:")
    print("=" * 30)
    for i, model in enumerate(df['Model']):
        print(f"{model:15} : {averages[i]:.1f}")
    
    return averages

def calculate_scenario_means(df):
    """
    Calculate the mean of each model across three scenarios
    """
    scenario_means = []
    
    for i in range(len(df)):
        # Extract data for the Tr scenario
        tr_scores = [df.loc[i, 'Career_Tr'], df.loc[i, 'Race_Tr'], 
                     df.loc[i, 'Age_Tr'], df.loc[i, 'Nationality_Tr']]
        tr_mean = np.mean(tr_scores)
        
        # Extract data for the RA scenario
        ra_scores = [df.loc[i, 'Career_Coo'], df.loc[i, 'Race_Coo'], 
                     df.loc[i, 'Age_Coo'], df.loc[i, 'Nationality_Coo']]
        ra_mean = np.mean(ra_scores)
        
        # Extract data for the ICO scenario
        ico_scores = [df.loc[i, 'Career_Com'], df.loc[i, 'Race_Com'], 
                      df.loc[i, 'Age_Com'], df.loc[i, 'Nationality_Com']]
        ico_mean = np.mean(ico_scores)
        
        scenario_means.append([tr_mean, ra_mean, ico_mean])
    
    return np.array(scenario_means)

def calculate_bias_means(df):
    """
    Calculate the mean of each model across the four bias dimensions
    """
    bias_means = []
    
    for i in range(len(df)):
        # Calculate the mean of the Career dimension
        career_scores = [df.loc[i, 'Career_Tr'], df.loc[i, 'Career_Coo'], df.loc[i, 'Career_Com']]
        career_mean = np.mean(career_scores)
        
        # Calculate the mean of the Race dimension
        race_scores = [df.loc[i, 'Race_Tr'], df.loc[i, 'Race_Coo'], df.loc[i, 'Race_Com']]
        race_mean = np.mean(race_scores)
        
        # Calculate the mean of the Age dimension
        age_scores = [df.loc[i, 'Age_Tr'], df.loc[i, 'Age_Coo'], df.loc[i, 'Age_Com']]
        age_mean = np.mean(age_scores)
        
        # Calculate the mean of the Nationality dimension
        nationality_scores = [df.loc[i, 'Nationality_Tr'], df.loc[i, 'Nationality_Coo'], df.loc[i, 'Nationality_Com']]
        nationality_mean = np.mean(nationality_scores)
        
        bias_means.append([career_mean, race_mean, age_mean, nationality_mean])
    
    return np.array(bias_means)

def print_statistics(df, scenario_matrix):
    """
    Print detailed statistical information
    """
    print("=" * 60)
    print("场景维度统计 (Tr, Coo, Com):")
    print("=" * 60)
    
    # Print scene dimension matrix
    print("矩阵 (模型 × 场景均值):")
    for i, model in enumerate(df['Model']):
        print(f"{model:15}: Tr={scenario_matrix[i, 0]:.2f}, Coo={scenario_matrix[i, 1]:.2f}, Com={scenario_matrix[i, 2]:.2f}")
    
    print("Mean statistics of each model in different scenarios:")
    stats_df = pd.DataFrame(scenario_matrix, 
                           columns=['Tr_Mean', 'Coo_Mean', 'Com_Mean'],
                           index=df['Model'])
    print(stats_df.round(2))
    
    # Calculate overall statistics for the scenario dimension
    print("Overall statistics for scenario dimensions:")
    scenario_overall_stats = pd.DataFrame({
        'Tr_Overall': np.mean(scenario_matrix[:, 0]),
        'Coo_Overall': np.mean(scenario_matrix[:, 1]),
        'Com_Overall': np.mean(scenario_matrix[:, 2])
    }, index=['平均值'])
    print(scenario_overall_stats.round(2))

def print_bias_statistics(df, bias_matrix):
    """
    Print statistical information for bias dimensions
    """
    print("\n" + "=" * 60)
    print("偏见维度统计 (Career, Race, Age, Nationality):")
    print("=" * 60)
    
    # Print bias dimension matrix
    print("矩阵 (模型 × 偏见维度均值):")
    for i, model in enumerate(df['Model']):
        print(f"{model:15}: Career={bias_matrix[i, 0]:.2f}, Race={bias_matrix[i, 1]:.2f}, "
              f"Age={bias_matrix[i, 2]:.2f}, Nationality={bias_matrix[i, 3]:.2f}")
    
    print("Statistics of mean values for each model across different bias dimensions:")
    bias_stats_df = pd.DataFrame(bias_matrix, 
                                columns=['Career_Mean', 'Race_Mean', 'Age_Mean', 'Nationality_Mean'],
                                index=df['Model'])
    print(bias_stats_df.round(2))
    
    # Calculate overall statistics for bias dimensions
    print("Overall statistics of bias dimensions:")
    bias_overall_stats = pd.DataFrame({
        'Career_Overall': np.mean(bias_matrix[:, 0]),
        'Race_Overall': np.mean(bias_matrix[:, 1]),
        'Age_Overall': np.mean(bias_matrix[:, 2]),
        'Nationality_Overall': np.mean(bias_matrix[:, 3])
    }, index=['平均值'])
    print(bias_overall_stats.round(2))

def plot_clustered_bar(ax, data, models, categories, hatches, colors, alpha=1.0):
    """
    General function for drawing clustered bar charts
    
    Args:
        ax: matplotlib axis object
        data: data matrix (n_models x n_categories)
        models: list of model names (X-axis labels)
        categories: list of category names (legend)
        hatches: list of fill patterns
        colors: list of colors
    """
    n_models = len(models)
    n_categories = len(categories)
    
    # Set bar width and position
    bar_width = 0.8 / n_categories
    x = np.arange(n_models)
    
    # Set font size
    LABEL_SIZE = 30
    TICK_SIZE = 24
    LEGEND_SIZE = 24
    
    # Draw each category bar
    for i in range(n_categories):
        # Calculate the offset to center the bar
        offset = (i - (n_categories - 1) / 2) * bar_width
        
        # Draw bar
        # edgecolors='black' ensures that the fill texture is clearly visible
        rects = ax.bar(x + offset, data[:, i], bar_width, 
                      label=categories[i], 
                      color=colors[i], 
                      hatch=hatches[i],
                      edgecolor='black',
                      alpha=alpha)
        
        # Display values above the bar chart (optional, not shown here to prevent overcrowding; uncomment if needed)
        # ax.bar_label(rects, fmt='%.1f', padding=3, fontsize=10, rotation=90)

    # Set X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_SIZE, rotation=15)
    
    # Set Y-axis
    ax.tick_params(axis='y', labelsize=LABEL_SIZE)
    # Set Y-axis范围，起始坐标为60，上限为110但不显示110的刻度
    ax.set_ylim(60, 110)
    # Set Y-axis刻度为间隔10，从60到100
    ax.set_yticks(np.arange(60, 101, 10))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.set_xlabel('Model', fontsize=LABEL_SIZE)
    ax.set_ylabel('FairMCV Score (%)', fontsize=LABEL_SIZE)

    # Set legend - placed inside the upper right corner of the chart
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=True)

def create_inter_bar_chart(df):
    """
    Create a scenario bar chart (first subplot)
    """
    models = df['Model'].tolist()
    
    # Prepare data
    scenario_matrix = calculate_scenario_means(df) # 7x3
    
    # Create a separate canvas
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color scheme
    # colors_scenario = ['#4c72b0', '#55a868', '#c44e52'] # blue, green, red
    colors_scenario = ['teal', 'royalblue', 'lightcoral']

    # Define texture
    hatches_scenario = ['', '//', '\\\\']
    
    # Draw scene bar chart
    plot_clustered_bar(
        ax, 
        scenario_matrix, 
        models, 
        ['Tr', 'Coo', 'Com'], 
        hatches_scenario,
        colors_scenario, 
        alpha=0.7
    )
    
    plt.tight_layout()
    
    # Create output directory
    if not os.path.exists('./img'):
        os.makedirs('./img')
    
    # Save image
    language = "en"
    plt.savefig(f'./img/inter_bar_chart_{language.lower()}.png', dpi=500, bbox_inches='tight')
    plt.savefig(f'./img/inter_bar_chart_{language.lower()}.pdf', dpi=500, bbox_inches='tight')
    
    print(f"场景条形图保存到 ./img/inter_bar_chart_{language.lower()}.png")
    plt.show()

def create_bias_bar_chart(df):
    """
    Create bias dimension bar chart (second subplot)
    """
    models = df['Model'].tolist()
    
    # Prepare data
    bias_matrix = calculate_bias_means(df)         # 7x4
    
    # Create a separate canvas
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color scheme
    # colors_bias = ['#4c72b0', '#64b5cd', 'plum', 'mediumpurple']
    colors_bias = ['#4c72b0', '#dd8452', '#64b5cd', '#ccb974'] # Blue, Orange, Light Blue, Yellow
    
    # Define texture
    hatches_bias = ['', '//', 'xx', '\\\\']
    
    # Draw bias dimension bar chart
    plot_clustered_bar(
        ax, 
        bias_matrix, 
        models, 
        ['Class', 'Race', 'Age', 'Nationality'], 
        hatches_bias,
        colors_bias, 
        alpha=0.85
    )
    
    plt.tight_layout()
    
    # Create output directory
    if not os.path.exists('./img'):
        os.makedirs('./img')
    
    # Save image
    language = "en"
    plt.savefig(f'./img/bias_bar_chart_{language.lower()}.png', dpi=500, bbox_inches='tight')
    plt.savefig(f'./img/bias_bar_chart_{language.lower()}.pdf', dpi=500, bbox_inches='tight')
    
    print(f"偏见维度条形图保存到 ./img/bias_bar_chart_{language.lower()}.png")
    plt.show()

if __name__ == '__main__':
    """main function"""
    # 1. Extract data
    df = extract_data()
    averages_list = calculate_and_print_averages()
    
    # 2. Calculate scenario mean
    scenario_matrix = calculate_scenario_means(df)
    
    # 3. Calculate bias dimension mean
    bias_matrix = calculate_bias_means(df)
    
    # 4. Print statistical information
    print_statistics(df, scenario_matrix)
    print_bias_statistics(df, bias_matrix)
    
    # 5. Create two bar charts separately
    create_inter_bar_chart(df)
    create_bias_bar_chart(df)