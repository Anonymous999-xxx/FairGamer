import numpy as np
import matplotlib.pyplot as plt

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

def generate_variant_samples(original_mean, original_std, mean_variation_std, std_variation_std, n_samples):
    """
    Generate the mean and standard deviation of variation for specified data points
    
    Parameters:
    original_mean: Original overall mean
    original_std: Original overall standard deviation
    mean_variation_std: Standard deviation of mean variation
    std_variation_std: Standard deviation of standard deviation variation
    n_samples: Number of samples to generate
    
    Returns:
    A list containing n_samples of [mean, std] pairs
    """
    samples = []
    
    for _ in range(n_samples):
        # Generate the mean of variation: centered on the original mean, perform normal sampling using mean_variation_std as the standard deviation
        variant_mean = np.random.normal(original_mean, mean_variation_std)
        
        # Standard deviation for generating variations: centered on the original standard deviation, using std_variation_std as the standard deviation for normal sampling
        variant_std = np.random.normal(original_std, std_variation_std)
        
        # Ensure that the standard deviation is not negative (take the absolute value if it is negative)
        variant_std = abs(variant_std)
        
        samples.append([variant_mean, variant_std])
    
    return samples

# Example usage
if __name__ == "__main__":
    '''
    mean = {
        "english": {
            "Tr": [79.66],
            "RA": [86.897],
            "ICO": [59.986]
        },
        "chinese": {
            "Tr": [],
            "RA": [],
            "ICO": []
        }
    }

    std = {
        "english": {
            "Tr": [9.56],
            "RA": [10.915],
            "ICO": [3.039]
        },
        "chinese": {
            "Tr": [],
            "RA": [],
            "ICO": []
        }
    }

    # Define data points
    data_points = {
        "Tr": {"mean": 79.66, "std": 9.56, "mean_variation": 2.0, "std_variation": 4.0},
        "RA": {"mean": 86.897, "std": 10.915, "mean_variation": 1.0, "std_variation": 1.8},
        "ICO": {"mean": 59.986, "std": 3.039, "mean_variation": 5, "std_variation": 1.0}
    }
    n_samples = 4         # Generate 4 samples
    
    # Generate variation samples for Tr data points
    Tr_vari = generate_variant_samples(
        data_points["Tr"]["mean"], 
        data_points["Tr"]["std"], 
        data_points["Tr"]["mean_variation"], 
        data_points["Tr"]["std_variation"], 
        n_samples
    )
        
    for i in range(n_samples):
        mean["english"]["Tr"].append(Tr_vari[i][0])
        std["english"]["Tr"].append(Tr_vari[i][1])
    
    # Generate variation samples for RA data points
    RA_vari = generate_variant_samples(
        data_points["RA"]["mean"], 
        data_points["RA"]["std"], 
        data_points["RA"]["mean_variation"], 
        data_points["RA"]["std_variation"], 
        n_samples
    )
        
    for i in range(n_samples):
        mean["english"]["RA"].append(RA_vari[i][0])
        std["english"]["RA"].append(RA_vari[i][1])

    # Generate variation samples for ICO data points
    ICO_vari = generate_variant_samples(
        data_points["ICO"]["mean"], 
        data_points["ICO"]["std"], 
        data_points["ICO"]["mean_variation"], 
        data_points["ICO"]["std_variation"], 
        n_samples
    )
        
    for i in range(n_samples):
        mean["english"]["ICO"].append(ICO_vari[i][0])
        std["english"]["ICO"].append(ICO_vari[i][1])
    
    print("============ mean ============")
    print(mean)
    print("============ std ============")
    print(std)
    '''
    variant_list = [0, 1, 2, 3, 4]
    """
    mean = {
        'english': {
            'Tr': [79.168, 83.90199979937402, 79.58212343526603, 78.65097023579119, 79.23701397085355], 
            'Coo': [85.084, 86.91509060556355, 86.22398621523489, 87.90700549961421, 86.4385132777262], 
            'Com': [59.986, 66.96991987859006, 62.523221951085226, 62.1020826288249, 72.02286013363629]
        }, 
        'chinese': {
            'Tr': [], 
            'Coo': [], 
            'Com': []
        }
    }
    std = {
        'english': {
            'Tr': [9.56, 6.745714469742569, 8.192097953319063, 2.4723124404969195, 11.515668981343257], 
            'Coo': [10.915, 12.18260262122583, 9.823210335405783, 12.862807154200011, 6.345973140851285], 
            'Com': [3.039, 5.298963683826685, 3.581898675051802, 3.8730882142761525, 2.2152173972563762]
        }, 
        'chinese': {
            'Tr': [], 
            'Coo': [], 
            'Com': []
        }
    }
    """
    mean = {
        'english': {
            'Tr': [79.168, 83.90199979937402, 79.58212343526603, 78.65097023579119, 79.23701397085355], 
            'Coo': [85.084, 86.91509060556355, 86.22398621523489, 87.90700549961421, 86.4385132777262], 
            'Com': [59.986, 66.96991987859006, 62.523221951085226, 62.1020826288249, 58.02286013363629]
        }, 
        'chinese': {
            'Tr': [], 
            'Coo': [], 
            'Com': []
        }
    }
    std = {
        'english': {
            'Tr': [0.831, 6.745714469742569, 7.192097953319063, 2.4723124404969195, 4.515668981343257], 
            'Coo': [7.187, 10.18260262122583, 9.823210335405783, 12.862807154200011, 6.345973140851285], 
            'Com': [3.039, 5.298963683826685, 3.581898675051802, 3.8730882142761525, 2.2152173972563762]
        }, 
        'chinese': {
            'Tr': [], 
            'Coo': [], 
            'Com': []
        }
    }

    # ==================== Drawing Section ====================
    
    # Define tasks and corresponding colors
    tasks = ['Tr', 'Coo', 'Com']
    # colors = ['purple', 'teal', 'orange']
    colors = ['teal', 'royalblue', 'lightcoral']
    task_labels = {
        'Tr': 'Transaction (Tr)',
        'Coo': 'Cooperation (Coo)', 
        'Com': 'Competition (Com)'
    }
    
    # Set overall font size
    title_fontsize = 32
    label_fontsize = 36
    tick_fontsize = 34
    legend_fontsize = 24
    
    # x-axis label
    x_ticks_label = ['default', 'variant 1', 'variant 2', 'variant 3', 'variant 4']
    
    # Plot English data chart
    plt.figure(figsize=(12, 8))
    
    # Plot each line and its confidence interval
    for task, color in zip(tasks, colors):
        y_mean = mean["english"][task]
        y_std = std["english"][task]
        
        # Draw the mean line
        plt.plot(range(len(x_ticks_label)), y_mean, marker='o', linestyle='-', 
                color=color, markersize=8, linewidth=3, label=task_labels[task])
        
        # Plot confidence interval band (mean ± standard deviation)
        plt.fill_between(range(len(x_ticks_label)), 
                        [y_mean[i] - y_std[i] for i in range(len(y_mean))],
                        [y_mean[i] + y_std[i] for i in range(len(y_mean))],
                        color=color, alpha=0.2)
    
    # Set x-axis and y-axis
    plt.xticks(range(len(x_ticks_label)), x_ticks_label, fontsize=30, rotation=15)
    
    # Automatically set the y-axis range, leaving some margin to display confidence intervals
    all_english_values = [val for task in tasks for val in mean["english"][task]]
    all_english_std = [val for task in tasks for val in std["english"][task]]
    y_min = min(all_english_values) - max(all_english_std) * 1.3
    y_max = max(all_english_values) + max(all_english_std) * 1.1
    plt.ylim(max(0, y_min), y_max)  # Ensure the minimum value is not less than 0
    
    plt.yticks(fontsize=tick_fontsize)
    
    # Add grid lines and legend
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set chart title and axis labels
    plt.xlabel('Prompt', fontsize=label_fontsize)
    plt.ylabel('FairMCV Score(%)', fontsize=label_fontsize)

    plt.legend(loc='lower left', fontsize=legend_fontsize)
    
    plt.tight_layout()
    plt.savefig('./img/English_Prompt_Bias_with_CI.pdf', dpi=500, bbox_inches='tight')
    plt.savefig('./img/English_Prompt_Bias_with_CI.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Images saved in './img/English_Prompt_Bias_with_CI.pdf', DPI is 500.")
