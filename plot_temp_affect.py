import matplotlib.pyplot as plt

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

if __name__ == '__main__':
    en_tempature = {
        'GGS-Real': [0.525, 0.505, 0.472,0.489, 0.451, 0.440],
        'GGS-Virtual': [0.208, 0.187, 0.172, 0.161, 0.164, 0.155],
        'SNPC-Real': [0.487, 0.506, 0.382, 0.301, 0.281, 0.237],
        'SNPC-Virtual': [0.382, 0.366, 0.312, 0.298, 0.267, 0.195],
        'ICO-Real': [1.681, 0.765, 0.814, 0.514, 0.327, 0.171],
        'ICO-Virtual': [2.349, 1.912, 1.158, 0.798, 0.412, 0.293],
    }
    ch_tempature = {
        'GGS-Real': [0.399, 0.417, 0.392, 0.349, 0.321, 0.309],
        'GGS-Virtual': [0.308, 0.273, 0.276, 0.264, 0.243, 0.236],
        'SNPC-Real': [0.427, 0.376, 0.342, 0.281, 0.221, 0.196],
        'SNPC-Virtual': [0.401, 0.379, 0.348, 0.264, 0.197, 0.165],
        'ICO-Real': [1.411, 1.099, 0.854, 0.678, 0.482, 0.172],
        'ICO-Virtual': [1.953, 1.609, 0.989, 0.765, 0.321, 0.127],
    }
    
    # Define task order and corresponding colors
    tasks = ['SNPC-Real', 'SNPC-Virtual', 'ICO-Real', 'ICO-Virtual', 'GGS-Real', 'GGS-Virtual']
    colors = ['purple', 'royalblue', 'teal', 'lightcoral', 'orange', 'gold']

    # Set global font size
    title_fontsize = 32
    label_fontsize = 24
    tick_fontsize = 22
    legend_fontsize = 20
    
    # x-axis labels
    x_ticks_label = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Plot English data
    plt.figure(figsize=(10, 6))
    
    # Plot each line
    for task, color in zip(tasks, colors):
        plt.plot(en_tempature[task], marker='o', linestyle='-', color=color, markersize=8, linewidth=3, label=task)
    
    # Set chart title and axis labels
    # plt.title('English Data - Temperature vs D_lstd', fontsize=title_fontsize)
    plt.xlabel('Temperature', fontsize=label_fontsize)
    plt.ylabel(r"$D_{lstd}$", fontsize=label_fontsize)
    
    # Set x-axis and y-axis
    plt.xticks([0, 1, 2, 3, 4, 5], x_ticks_label, fontsize=tick_fontsize)
    plt.ylim(0, 2.5)  # Unify y-axis range to include all data
    plt.yticks(fontsize=tick_fontsize)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=legend_fontsize)
    
    plt.tight_layout()

    plt.savefig('./img/English_Temp_Affect.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot Chinese data
    plt.figure(figsize=(10, 6))
    
    # Plot each line
    for task, color in zip(tasks, colors):
        plt.plot(ch_tempature[task], marker='o', linestyle='-', color=color, markersize=8, linewidth=3, label=task)
    
    # Set chart title and axis labels
    # plt.title('Chinese Data - Temperature vs D_lstd', fontsize=title_fontsize)
    plt.xlabel('Temperature', fontsize=label_fontsize)
    plt.ylabel(r"$D_{lstd}$", fontsize=label_fontsize)
    
    # Set x-axis and y-axis
    plt.xticks([0, 1, 2, 3, 4, 5], x_ticks_label, fontsize=tick_fontsize)
    plt.ylim(0, 2.5)  # 统一y轴范围以包含所有数据
    plt.yticks(fontsize=tick_fontsize)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=legend_fontsize)
    
    plt.tight_layout()

    plt.savefig('./img/Chinese_Temp_Affect.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Images saved in './img/English_Temp_Affect.pdf' and './img/Chinese_Temp_Affect.pdf', DPI is 500.")
