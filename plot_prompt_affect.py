import matplotlib.pyplot as plt

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

if __name__ == '__main__':
    # 定义数据
    en_prompt = {
        'GGS-Real': [0.440, 0.471, 0.413, 0.511, 0.405],
        'GGS-Virtual': [0.155, 0.141, 0.163, 0.166, 0.140],
        'SNPC-Real': [0.237, 0.197, 0.262, 0.269, 0.221],
        'SNPC-Virtual': [0.195, 0.220, 0.209, 0.236, 0.175],
        'ICO-Real': [0.171, 0.186, 0.156, 0.224, 0.192],
        'ICO-Virtual': [0.293, 0.309, 0.297, 0.315, 0.352], 
    }

    ch_prompt = {
        'GGS-Real': [0.309, 0.320, 0.291, 0.346, 0.306],
        'GGS-Virtual': [0.236, 0.276, 0.274, 0.259, 0.213],
        'SNPC-Real': [0.196, 0.229, 0.221, 0.213, 0.140],
        'SNPC-Virtual': [0.165, 0.116, 0.181, 0.152, 0.167],
        'ICO-Real': [0.172, 0.167, 0.183, 0.165, 0.109],
        'ICO-Virtual': [0.127, 0.111, 0.086, 0.089, 0.120],
    }

    # 定义任务顺序和对应的颜色
    tasks = ['SNPC-Real', 'SNPC-Virtual', 'ICO-Real', 'ICO-Virtual', 'GGS-Real', 'GGS-Virtual']
    colors = ['purple', 'royalblue', 'teal', 'lightcoral', 'orange', 'gold']

    # 设置全局字体大小
    title_fontsize = 32
    label_fontsize = 24
    tick_fontsize = 22
    legend_fontsize = 20
    
    # x轴标签
    x_ticks_label = ['default', 'variant 1', 'variant 2', 'variant 3', 'variant 4']
    
    # 绘制英文数据图
    plt.figure(figsize=(10, 6))
    
    # 绘制每条折线
    for task, color in zip(tasks, colors):
        plt.plot(en_prompt[task], marker='o', linestyle='-', color=color, markersize=8, linewidth=3, label=task)
    
    # 设置图表标题和坐标轴标签
    # plt.title('English Data - Temperature vs D_lstd', fontsize=title_fontsize)
    plt.xlabel('Prompt', fontsize=label_fontsize)
    plt.ylabel(r"$D_{lstd}$", fontsize=label_fontsize)
    
    # 设置x轴和y轴
    plt.xticks([0, 1, 2, 3, 4], x_ticks_label, fontsize=tick_fontsize)
    plt.ylim(0, 0.8)  # 统一y轴范围以包含所有数据
    plt.yticks(fontsize=tick_fontsize)
    
    # 添加网格线和图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=legend_fontsize, loc='upper right')
    
    plt.tight_layout()

    plt.savefig('./img/English_Prompt_Affect.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 绘制中文数据图
    plt.figure(figsize=(10, 6))
    
    # 绘制每条折线
    for task, color in zip(tasks, colors):
        plt.plot(ch_prompt[task], marker='o', linestyle='-', color=color, markersize=8, linewidth=3, label=task)
    
    # 设置图表标题和坐标轴标签
    # plt.title('Chinese Data - Temperature vs D_lstd', fontsize=title_fontsize)
    plt.xlabel('Prompt', fontsize=label_fontsize)
    plt.ylabel(r"$D_{lstd}$", fontsize=label_fontsize)
    
    # 设置x轴和y轴
    plt.xticks([0, 1, 2, 3, 4], x_ticks_label, fontsize=tick_fontsize)
    plt.ylim(0, 0.8)  # 统一y轴范围以包含所有数据
    plt.yticks(fontsize=tick_fontsize)
    
    # 添加网格线和图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=legend_fontsize, loc='upper right')
    
    plt.tight_layout()

    plt.savefig('./img/Chinese_Prompt_Affect.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Images saved in './img/English_Prompt_Affect.pdf' and './img/Chinese_Prompt_Affect.pdf', DPI is 500.")
