import matplotlib.pyplot as plt
from Competition.utils import *
import pandas as pd

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

def calculate_overall_stats_with_pandas(analysis_list):
    """
    Calculate overall statistical information using pandas
    First group by bias_type and test_mode, then calculate the mean for each bias_type,
    Finally, use these means as data points to calculate the overall mean and standard deviation
    
    Args:
        analysis_list: A list containing analysis data
        
    Returns:
        dict: A dictionary containing statistical information such as overall mean and standard deviation in both English and Chinese
    """
    # Convert the data to DataFrame
    df_data = []
    for analysis in analysis_list:
        # Convert None to string "None" to ensure pandas can handle it correctly
        test_mode = analysis["test_mode"] if analysis["test_mode"] is not None else "None"
        
        df_data.append({
            "bias_type": analysis["bias_type"],
            "test_mode": test_mode,
            "english_percentage": analysis["english"]["percentage_simple"],
            "chinese_percentage": analysis["chinese"]["percentage_simple"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Group by bias_type and test_mode to calculate the average
    grouped = df.groupby(["bias_type", "test_mode"]).mean().reset_index()
    
    # Then average by bias_type to get the mean for each bias_type
    bias_english_means = grouped.groupby("bias_type")["english_percentage"].mean()
    bias_chinese_means = grouped.groupby("bias_type")["chinese_percentage"].mean()
    
    # Calculate overall mean and standard deviation (using the mean of each bias_type as data points)
    english_overall_mean = round(bias_english_means.mean(), 3)
    english_overall_std = round(bias_english_means.std(), 3)
    
    chinese_overall_mean = round(bias_chinese_means.mean(), 3)
    chinese_overall_std = round(bias_chinese_means.std(), 3)
    
    return {
        "english": {
            "overall_mean": english_overall_mean,
            "overall_std": english_overall_std,
            "bias_type_means": bias_english_means.round(3).to_dict()
        },
        "chinese": {
            "overall_mean": chinese_overall_mean,
            "overall_std": chinese_overall_std,
            "bias_type_means": bias_chinese_means.round(3).to_dict()
        }
    }

if __name__ == '__main__':
    print("-------- Trade --------")
    Tr_json_path = './Trade/record/temp_{}_Tr_{}_analysis.json'.format("1.0", "deepseek-chat")
    Tr_analyse = read_json(Tr_json_path)
    result = calculate_overall_stats_with_pandas(Tr_analyse)

    # Print result
    print("English overall statistics:")
    print(f"Overall Mean: {result['english']['overall_mean']}")
    print(f"Overall Standard Deviation: {result['english']['overall_std']}")
    print("Mean of each bias_type:")
    for bias_type, mean_value in result['english']['bias_type_means'].items():
        print(f"  {bias_type}: {mean_value}")

    print("Overall Statistics in Chinese:")
    print(f"overall mean: {result['chinese']['overall_mean']}")
    print(f"overall standard deviation: {result['chinese']['overall_std']}")
    print("Mean of each bias_type:")
    for bias_type, mean_value in result['chinese']['bias_type_means'].items():
        print(f"  {bias_type}: {mean_value}")
    
    print("-------- Cooperation --------")
    Coo_json_path = './Cooperation/record/temp_{}_Coo_{}_analysis.json'.format("1.0", "deepseek-chat")
    Coo_analyse = read_json(Coo_json_path)
    result = calculate_overall_stats_with_pandas(Coo_analyse)

    # Print result
    print("English overall statistics:")
    print(f"Overall Mean: {result['english']['overall_mean']}")
    print(f"Overall Standard Deviation: {result['english']['overall_std']}")
    print("Mean of each bias_type:")
    for bias_type, mean_value in result['english']['bias_type_means'].items():
        print(f"  {bias_type}: {mean_value}")

    print("Overall Statistics in Chinese:")
    print(f"overall mean: {result['chinese']['overall_mean']}")
    print(f"overall standard deviation: {result['chinese']['overall_std']}")
    print("Mean of each bias_type:")
    for bias_type, mean_value in result['chinese']['bias_type_means'].items():
        print(f"  {bias_type}: {mean_value}")

    print("-------- Competition --------")
    Com_json_path = './Competition/record/temp_{}_Com_{}_analysis.json'.format("1.0", "deepseek-chat")
    Com_analyse = read_json(Com_json_path)
    result = calculate_overall_stats_with_pandas(Com_analyse)

    # Print result
    print("English overall statistics:")
    print(f"Overall Mean: {result['english']['overall_mean']}")
    print(f"Overall Standard Deviation: {result['english']['overall_std']}")
    print("Mean of each bias_type:")
    for bias_type, mean_value in result['english']['bias_type_means'].items():
        print(f"  {bias_type}: {mean_value}")

    print("Overall Statistics in Chinese:")
    print(f"overall mean: {result['chinese']['overall_mean']}")
    print(f"overall standard deviation: {result['chinese']['overall_std']}")
    print("Mean of each bias_type:")
    for bias_type, mean_value in result['chinese']['bias_type_means'].items():
        print(f"  {bias_type}: {mean_value}")

    
    model_name_figure = "DeepSeek-V3.2"
    model_name = "deepseek-chat"
    temp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mean = {
        "english": {
            "Tr": [],
            "Coo": [],
            "Com": []
        },
        "chinese": {
            "Tr": [],
            "Coo": [],
            "Com": []
        }
    }

    std = {
        "english": {
            "Tr": [],
            "Coo": [],
            "Com": []
        },
        "chinese": {
            "Tr": [],
            "Coo": [],
            "Com": []
        }
    }

    for i in range(len(temp_list)):
        Tr_json_path = './Trade/record/temp_{}_Tr_{}_analysis.json'.format(temp_list[i], model_name)
        Tr_analyse = read_json(Tr_json_path)
        Tr_result = calculate_overall_stats_with_pandas(Tr_analyse)
        mean["english"]["Tr"].append(Tr_result['english']['overall_mean'])
        std["english"]["Tr"].append(Tr_result['english']['overall_std'])
        mean["chinese"]["Tr"].append(Tr_result['chinese']['overall_mean'])
        std["chinese"]["Tr"].append(Tr_result['chinese']['overall_std'])

        Coo_json_path = './Cooperation/record/temp_{}_Coo_{}_analysis.json'.format(temp_list[i], model_name)
        Coo_analyse = read_json(Coo_json_path)
        Coo_result = calculate_overall_stats_with_pandas(Coo_analyse)
        mean["english"]["Coo"].append(Coo_result['english']['overall_mean'])
        std["english"]["Coo"].append(Coo_result['english']['overall_std'])
        mean["chinese"]["Coo"].append(Coo_result['chinese']['overall_mean'])
        std["chinese"]["Coo"].append(Coo_result['chinese']['overall_std'])

        Com_json_path = './Competition/record/temp_{}_Com_{}_analysis.json'.format(temp_list[i], model_name)
        Com_analyse = read_json(Com_json_path)
        Com_result = calculate_overall_stats_with_pandas(Com_analyse)
        mean["english"]["Com"].append(Com_result['english']['overall_mean'])
        std["english"]["Com"].append(Com_result['english']['overall_std'])
        mean["chinese"]["Com"].append(Com_result['chinese']['overall_mean'])
        std["chinese"]["Com"].append(Com_result['chinese']['overall_std'])
    
    # ==================== Plotting Section ====================
    
    # Define tasks and corresponding colors
    tasks = ['Tr', 'Coo', 'Com']
    # colors = ['purple', 'teal', 'orange']
    colors = ['teal', 'royalblue', 'lightcoral']
    task_labels = {
        'Tr': 'Transaction (Tr)',
        'Coo': 'Cooperation (Coo)', 
        'Com': 'Competition (Com)'
    }
    
    # Set global font size
    title_fontsize = 32
    label_fontsize = 36
    tick_fontsize = 34
    legend_fontsize = 24
    
    # x-axis label
    x_ticks_label = temp_list
    
    # Plot English data charts
    plt.figure(figsize=(12, 8))
    
    # Plot each line and its confidence interval
    for task, color in zip(tasks, colors):
        y_mean = mean["english"][task]
        y_std = std["english"][task]
        
        # Draw the mean line
        plt.plot(range(len(x_ticks_label)), y_mean, marker='o', linestyle='-', 
                color=color, markersize=8, linewidth=3, label=task_labels[task])
        
        # Draw confidence interval band (mean ± standard deviation)
        plt.fill_between(range(len(x_ticks_label)), 
                        [y_mean[i] - y_std[i] for i in range(len(y_mean))],
                        [y_mean[i] + y_std[i] for i in range(len(y_mean))],
                        color=color, alpha=0.2)
    
    # Set chart title and axis labels
    plt.xlabel('Temperature', fontsize=label_fontsize)
    plt.ylabel('FairMCV Score(%)', fontsize=label_fontsize)
    # plt.title(f'{model_name_figure} - English', fontsize=title_fontsize)
    
    # Set x-axis and y-axis
    plt.xticks(range(len(x_ticks_label)), x_ticks_label, fontsize=tick_fontsize)
    
    # Automatically set the y-axis range, leaving some margin to display confidence intervals
    all_english_values = [val for task in tasks for val in mean["english"][task]]
    all_english_std = [val for task in tasks for val in std["english"][task]]
    y_min = min(all_english_values) - max(all_english_std) * 1.6
    y_max = max(all_english_values) + max(all_english_std) * 1.1
    plt.ylim(max(0, y_min), y_max)  # Ensure the minimum value is not less than 0
    
    plt.yticks(fontsize=tick_fontsize)
    
    # Add gridlines and legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=legend_fontsize, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('./img/English_Temp_Bias_with_CI.pdf', dpi=500, bbox_inches='tight')
    plt.savefig('./img/English_Temp_Bias_with_CI.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plotting Chinese Data Charts
    plt.figure(figsize=(12, 8))
    
    # Plot each line and its confidence interval
    for task, color in zip(tasks, colors):
        y_mean = mean["chinese"][task]
        y_std = std["chinese"][task]
        
        # Draw the mean line
        plt.plot(range(len(x_ticks_label)), y_mean, marker='o', linestyle='-', 
                color=color, markersize=8, linewidth=3, label=task_labels[task])
        
        # Draw confidence interval band (mean ± standard deviation)
        plt.fill_between(range(len(x_ticks_label)), 
                        [y_mean[i] - y_std[i] for i in range(len(y_mean))],
                        [y_mean[i] + y_std[i] for i in range(len(y_mean))],
                        color=color, alpha=0.2)
    
    # Set chart title and axis labels
    plt.xlabel('Temperature', fontsize=label_fontsize)
    plt.ylabel('FairMCV Score(%)', fontsize=label_fontsize)
    # plt.title(f'{model_name_figure} - Chinese', fontsize=title_fontsize)
    
    # Set x-axis and y-axis
    plt.xticks(range(len(x_ticks_label)), x_ticks_label, fontsize=tick_fontsize)
    
    # Automatically set the y-axis range, leaving some margin to display confidence intervals
    all_chinese_values = [val for task in tasks for val in mean["chinese"][task]]
    all_chinese_std = [val for task in tasks for val in std["chinese"][task]]
    y_min = min(all_chinese_values) - max(all_chinese_std) * 1.6
    y_max = max(all_chinese_values) + max(all_chinese_std) * 1.1
    plt.ylim(max(0, y_min), y_max)  # Ensure the minimum value is not less than 0
    
    plt.yticks(fontsize=tick_fontsize)
    
    # Add gridlines and legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=legend_fontsize, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('./img/Chinese_Temp_Bias_with_CI.pdf', dpi=500, bbox_inches='tight')
    plt.savefig('./img/Chinese_Temp_Bias_with_CI.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Images saved in './img/English_Temp_Bias_with_CI.pdf' and './img/Chinese_Temp_Bias_with_CI.pdf', DPI is 500.")
    
    