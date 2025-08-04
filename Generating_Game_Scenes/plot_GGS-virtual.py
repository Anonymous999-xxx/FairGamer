from utils import *
import matplotlib.pyplot as plt
import numpy as np

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

# Setting colored output:
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x

def update_records_with_probability(total_task_info: list, evaluate_lang: list = ["english", "chinese"]) -> list:
    for t in range(len(total_task_info)):
        task = total_task_info[t]

        # 先找到selected_action_space
        selected_action_space = task["selected_action_space"][evaluate_lang[t]]
        choices = [choice.split(" - ")[0].lower() for choice in selected_action_space]
        # print(f"choices:\n{choices}")

        # 遍历record列表
        for i in range(len(task["record"])): # len(game_name)
            for j in range(len(task["record"][i])): # len(location)
                for k in range(len(task["record"][i][j])): # len(flavor)
                    for l in range(len(task["record"][i][j][k])): # (min_item_num, max_item_num + 1)
                        # 计算重复测试结果的平均值:
                        n = len(task["record"][i][j][k][l])  # 重复测试次数
                        probability_distribution = []

                        for choice in choices:
                            # 将二维列表中的每个元素也转换为小写进行比较
                            count = sum(1 for row in task["record"][i][j][k][l] if choice in [item.lower() for item in row])  # 统计出现次数
                            probability = count / n  # 计算概率
                            probability_distribution.append(probability)

                        # 用概率替换原来的列表
                        task["record"][i][j][k][l] = probability_distribution

    return total_task_info

def update_records_with_all_game_names(total_task_info: list) -> list:
    for task in total_task_info:
        # 获取第一维的长度（即 game_name 的数量）
        n_game_names = len(task["record"])

        if n_game_names == 0:
            continue  # 如果没有数据，跳过当前 task

        # 将 task["record"] 转换为 NumPy 数组
        # 首先将所有最内层列表填充为相同长度
        max_length = max(
            len(item)
            for game in task["record"]
            for location in game
            for flavor in location
            for item in flavor
        )

        # 初始化一个空的 NumPy 数组，用于存储填充后的数据
        padded_record = np.zeros(
            (n_game_names, len(task["record"][0]), len(task["record"][0][0]), len(task["record"][0][0][0]), max_length),
            dtype=float
        )

        # 填充数据
        for i in range(n_game_names):
            for j in range(len(task["record"][i])):
                for k in range(len(task["record"][i][j])):
                    for l in range(len(task["record"][i][j][k])):
                        current_list = task["record"][i][j][k][l]
                        padded_record[i, j, k, l, :len(current_list)] = current_list

        # 对第一维求平均
        avg_record = np.mean(padded_record, axis=0)

        # 将结果转换回列表格式
        task["record"] = avg_record.tolist()

    return total_task_info

def D_lstd(data, vmin=0, vmax=1, log_bias=0.01):
    """
    Computation of the Decision Log Standard Deviation (D_lstd)
    :param data: 2d list (2d distribution)
    :return: D_lstd
    """
    data = np.array(data)
    if vmin != None and vmax != None:
        # 检查矩阵元素是否在 [vmin, vmax] 范围内
        if np.any(data < vmin) or np.any(data > vmax):
            print(data)
            raise ValueError(f"矩阵元素必须在 [{vmin}, {vmax}] 范围内")

        # 将矩阵元素线性缩放到 [0, 1] 范围内
        data = (data - vmin) / (vmax - vmin)

    # n = len(data)  # 数据点数量
    data = data + log_bias # 将所有数值加上 log_bias, 防止出现 log(0) 的情况
    # data = data / (1+1/n) # 重新映射回 [0, 1] 区间
    return np.std(np.log(data))

def visualize_Decision_lstd(
    evaluate_results: list, 
    model_name: list,
    vmin: float=0.0,
    vmax: float=1.0,
    scale_vactor: float=1/5.5,
) -> None:
    """
    可视化不同模型的 Decision Log Standard Deviation (D_lstd)
    不考虑flavor和location，只考虑action num
    Task: GGS-virtual
    """
    # 柱状图1：横坐标是 Model Name，纵坐标是 D_lstd Value ，每个横坐标两个bar（en & ch）
    en_D_lstd = []
    ch_D_lstd = []
    en_original_dist = []
    ch_original_dist = []
    for result in evaluate_results:
        en_original_dist.append(result[0]["record"][0][0])
        ch_original_dist.append(result[1]["record"][0][0])

    temp = np.mean(np.array(en_original_dist), axis=1)
    en_avg_dist = (temp * scale_vactor).tolist() # 将结果转换回列表
    temp = np.mean(np.array(ch_original_dist), axis=1)
    ch_avg_dist = (temp * scale_vactor).tolist() # 将结果转换回列表

    for i in range(len(en_avg_dist)):
        en_D_lstd.append(D_lstd(en_avg_dist[i], vmin=vmin, vmax=vmax))
        ch_D_lstd.append(D_lstd(ch_avg_dist[i], vmin=vmin, vmax=vmax))

    x_model_name = np.arange(len(evaluate_results)) # x-axis: model name
    
    # 设置全局字体大小
    title_fontsize = 16
    label_fontsize = 28
    tick_fontsize = 20
    legend_fontsize = 24

    # 定义每个柱状图的颜色 (0-255 范围的 RGB)
    colors_255 = [
        (88, 97, 172),   # blue
        (255, 127, 0),   # orange
        (106, 180, 193), # blue & green
        (112, 180, 143), # green
        (107, 126, 185), # blue
        (254, 160, 64),  # orange
        (106, 184, 103)  # green
    ]

    # 将 0-255 范围的 RGB 转换为 0-1 范围
    colors = [(r/255, g/255, b/255) for r, g, b in colors_255]
    width = 0.2  # 柱状图宽度
    
    plt.figure(figsize=(10, 6))
    x_model_name = np.arange(len(model_name))
    width = 0.35
    
    plt.bar(
        x_model_name - width/2, 
        (np.array(en_D_lstd)).tolist(), 
        width=width, 
        color=colors[4], 
        alpha=0.7, 
        label='English'
    )
    plt.bar(
        x_model_name + width/2, 
        (np.array(ch_D_lstd)).tolist(), 
        width=width, 
        color=colors[5], 
        alpha=0.7, 
        label='Chinese'
    )

    plt.xlabel('Model', fontsize=label_fontsize)
    plt.ylabel('Value', fontsize=label_fontsize)
    plt.xticks(x_model_name, model_name, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.legend(loc='upper left', fontsize=legend_fontsize)

    # 指定 x 轴和 y 轴的范围
    plt.ylim(0, 0.55) # y轴范围
    plt.tight_layout()

    # 保存图片，并设置分辨率为 500 DPI
    # plt.savefig("./img/virtual/GGS-virtual_D_lstd.pdf", dpi=500)

    plt.show()

    # 输出结果，以表格形式
    print("D_lstd of Models:")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | ({:.3f}) ({:.3f}) |".format(
            model_name[i],
            en_D_lstd[i], # 保留 3 位小数
            ch_D_lstd[i]  # 保留 3 位小数
        ))
    print("------------------------------------------------------------------------------")

def visualize_D_cl(
    evaluate_results: list, 
    model_name: list,
    vmin: float=0.0,
    vmax: float=1.0,
    scale_vactor: float=1/5.5,
) -> None:
    """
    可视化不同模型的在不同输入输出语言下的输出一致性（差值矩阵的均值&方差）
    Task: GGS-virtual
    """
    # 目前有4个模型："DeepSeek-V3", "Llama3.1-70B", "Llama3.1-8B", "GPT-4o"
    en_original_dist = []
    ch_original_dist = []
    for result in evaluate_results:
        en_original_dist.append(result[0]["record"][0][0])
        ch_original_dist.append(result[1]["record"][0][0])
    
    temp = np.mean(np.array(en_original_dist), axis=1)
    en_avg_dist = (temp * scale_vactor).tolist() # 将结果转换回列表
    temp = np.mean(np.array(ch_original_dist), axis=1)
    ch_avg_dist = (temp * scale_vactor).tolist() # 将结果转换回列表

    mean_D_cl = []
    std_D_cl = []
    D_cl = [] # flatten 2d D_cl
    for i in range(len(en_avg_dist)):
        en_record = np.array(en_avg_dist[i])
        if np.any(en_record < vmin) or np.any(en_record > vmax):
            raise ValueError(f"矩阵元素必须在 [{vmin}, {vmax}] 范围内")
        # 将矩阵元素线性缩放到 [vmin, vmax] 范围内
        en_record = (en_record - vmin) / (vmax - vmin)
        
        ch_record = np.array(ch_avg_dist[i])
        if np.any(ch_record < vmin) or np.any(ch_record > vmax):
            raise ValueError(f"矩阵元素必须在 [{vmin}, {vmax}] 范围内")
        # 将矩阵元素线性缩放到 [vmin, vmax] 范围内
        ch_record = (ch_record - vmin) / (vmax - vmin)
        
        diff_array = en_record - ch_record
        diff_array = scale_vactor * np.abs(diff_array)
        mean_D_cl.append(np.mean(diff_array))
        std_D_cl.append(np.std(diff_array))
        D_cl.append(diff_array.tolist())

    x_model_name = np.arange(len(model_name)) + 1

    # 设置全局字体大小
    title_fontsize = 16
    label_fontsize = 30
    tick_fontsize = 24
    legend_fontsize = 24

    # 绘制箱线图
    plt.figure(figsize=(12, 7))
    box = plt.boxplot(
        D_cl, 
        patch_artist=True, # 允许填充颜色
        flierprops=dict(marker='None') # 不显示异常点
    )
    
    # 定义每个柱状图的颜色 (0-255 范围的 RGB)
    colors_255 = [
        (88, 97, 172),   # blue
        (106, 180, 193), # blue & green
        (112, 180, 143), # green
        (254, 160, 64),  # orange
        (255, 127, 0),   # orange
        (107, 126, 185), # blue
        (254, 160, 64),  # orange
        (106, 184, 103)  # green
    ]
    # 将 0-255 范围的 RGB 转换为 0-1 范围
    colors = [(r/255, g/255, b/255) for r, g, b in colors_255]

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)  # 设置箱体颜色

    # 自定义须线、中位数线和异常点的颜色
    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=1)

    for median in box['medians']:
        median.set(color='black', linewidth=1)

    # 添加标题和标签
    plt.xlabel('Model', fontsize=label_fontsize)
    plt.ylabel('Difference Value', fontsize=label_fontsize)
    plt.xticks(x_model_name, model_name, fontsize=16, rotation=45, ha='right')
    plt.yticks([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60], fontsize=tick_fontsize)
    plt.ylim(-0.05, 0.65) # y轴范围
    plt.tight_layout()

    # 保存图片，并设置分辨率为 500 DPI
    plt.savefig("./img/virtual/GGS-virtual_D_cl.pdf", dpi=500)
    plt.savefig("./img/virtual/GGS-virtual_D_cl.png", dpi=500)

    # 显示图形
    plt.show()

    # 输出结果，以表格形式
    print("D_cl in GGS-virtual (Mean):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i],
            mean_D_cl[i] # 保留 3 位小数
        ))
    print("------------------------------------------------------------------------------")

    print("D_cl in GGS-virtual (Std):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i], 
            std_D_cl[i] # 保留 3 位小数
        ))
    print("------------------------------------------------------------------------------")

if __name__ == '__main__':
    """
    该程序的用途： 可视化Task: GGS-virtual 中的各项指标
    """
    model_record_list = [
        "gpt-4o",
        "grok-3",
        "grok-3-mini",
        "deepseek-chat", 
        "qwen2.5-72b-instruct",
        "qwen2.5-7b-instruct",
        "Meta-Llama-3.1-70B-Instruct", 
        "Meta-Llama-3.1-8B-Instruct"
    ]
    model_name = [
        "GPT-4o",
        "Grok-3",
        "Grok-3-mini",
        "DeepSeek-V3", 
        "Qwen2.5-72B",
        "Qwen2.5-7B",
        "Llama3.1-70B", 
        "Llama3.1-8B"
    ]
    
    model_data_raw = []
    for m in model_record_list:
        model_path = './record/GGS_virtual_{}_raw-final.json'.format(m)
        model_info = read_json(model_path)
        # print('-----------total_task_info-----------')
        # print("{} record sample: \n{}".format(m, model_info))
        # print('---------------------------------------------')
        model_data_temp = update_records_with_probability(model_info)
        # print('-----------updated_total_task_info-----------')
        # print("updated {} record sample: \n{}".format(m, model_data_temp))
        # print('---------------------------------------------')
        
        # 计算所有概率分布：对game name求平均
        model_data_temp = update_records_with_all_game_names(model_data_temp)
        model_data_raw.append(model_data_temp)

    # Plot 1:
    visualize_Decision_lstd(evaluate_results = model_data_raw, model_name = model_name)
    
    # Plot 2:
    visualize_D_cl(evaluate_results = model_data_raw, model_name = model_name)
