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
    """
    对重复测试的n_r个结果求平均；
    对每个action_num的概率分布求平均
    """
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
                    item_count = [0]*len(choices)  # 用于存储每个item的出现次数
                    n_r = len(task["record"][i][j][k][0])  # 重复测试次数n_r
                    test_num = len(task["record"][i][j][k])  # action_num的种类
                    for l in range(len(task["record"][i][j][k])): # (min_item_num, max_item_num + 1)
                        for c_i in range(len(choices)):
                            count = sum(1 for row in task["record"][i][j][k][l] if choices[c_i] in [item.lower() for item in row])  # 统计出现次数
                            item_count[c_i] += count  # 累加每个item的出现次数
                    item_count = np.array(item_count) / (n_r * test_num)  # 计算每个item的概率
                    # 用概率替换原来的列表
                    task["record"][i][j][k] = item_count.tolist()

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
            len(flavor)
            for game in task["record"]
            for location in game
            for flavor in location
        )

        # 初始化一个空的 NumPy 数组，用于存储填充后的数据
        padded_record = np.zeros(
            (n_game_names, len(task["record"][0]), len(task["record"][0][0]), max_length),
            dtype=float
        )

        # 填充数据
        for i in range(n_game_names):
            for j in range(len(task["record"][i])):
                for k in range(len(task["record"][i][j])):
                    current_list = task["record"][i][j][k]
                    padded_record[i, j, k, :len(current_list)] = current_list

        # 对第一维求平均
        avg_record = np.mean(padded_record, axis=0)

        # 将结果转换回列表格式
        task["record"] = avg_record.tolist()

    return total_task_info

def count_labels(
    data: list, # 2d list
    labels: list = [0]*10+[1]*10+[2]*10+[3]*10+[4]*10, # 1d list
) -> list:
    """Get a normalized label distirbution from a 2d list of 50 items."""
    all_label_p = []
    for i in range(len(data)):
        current_list = data[i] # 1d distribution list

        # 计算标签分布
        temp = np.zeros(5)  # label 从 0 到 4
        for idx, prob in enumerate(current_list):
            label = labels[idx]
            temp[label] += prob
        all_label_p.append((np.array(temp) / np.sum(temp)).tolist())
    return all_label_p

def visualize_distribution(
    evaluate_results: list, 
    model_name: list,
    items: list, 
    scale_vactor: float= 1/5.5,
) -> None:
    """
    展示Z的分布情况 (DeepSeek-V3) ，以及各模型的文化偏好
    """
    en_original_dist = []
    ch_original_dist = []
    for result in evaluate_results:
        en_original_dist.append(result[0]["record"][0][0])
        ch_original_dist.append(result[1]["record"][0][0])
    
    en_scaled_dist = (np.array(en_original_dist) * scale_vactor).tolist() # scaling
    ch_scaled_dist = (np.array(ch_original_dist) * scale_vactor).tolist() # scaling

    # 获取 selected_action_space 的长度
    n_choices = len(items)
    labels = result[0]["selected_action_space"]["label"] # Get labels
    en_label_dist = count_labels(en_scaled_dist, labels)
    ch_label_dist = count_labels(ch_scaled_dist, labels)

    location_xticklabels = [
        "European &   \nNorth American", 
        "East Asian", 
        "Southeast Asian",
        "South Asian",
        "Central &   \nSouthern African"
    ]

    # 设置全局字体大小
    title_fontsize = 18
    label_fontsize = 14 # 14
    tick_fontsize = 10 # 8
    legend_fontsize = 10

    # 创建画布，设置子图布局
    fig, axes = plt.subplots(
        2, 2, 
        figsize=(14, 9),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.2, "hspace": 1.30}  # 增加子图之间的水平间距
    )

    # 调整子图的布局
    
    plt.subplots_adjust(
        left=0.05,     # 左边距
        right=0.95,   # 右边距
        #top=0.95,      # 上边距（增加顶部空间）
        bottom=0.23,  # 下边距（增加底部空间）
        #wspace=0.2,   # 水平间距
        #hspace=1.2    # 垂直间距
    )
    
    # 绘制概率分布柱状图a1
    ax = axes[0, 0]
    ax.bar(range(n_choices), en_scaled_dist[3], width=0.3, color='blue', alpha=0.7)
    ax.set_title(f"(a1) Output Distribution of DeepSeek-V3 w/ English", fontsize=title_fontsize)
    ax.set_xlabel("Item Name", fontsize=label_fontsize)
    ax.set_ylabel("Probability", fontsize=label_fontsize)
    ax.set_xticks(range(n_choices))
    ax.set_yticks([0, 0.05, 0.10])
    ax.set_xticklabels([item.split(" - ")[0] for item in items], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(0, 0.12) # y轴范围

    # 绘制标签分布柱状图a2
    ax = axes[0, 1]
    ax.bar(range(5), en_label_dist[3], width=0.2, color='#FEA040', alpha=0.7)
    ax.set_title("(a2) Accumulative Probability w/ English", fontsize=title_fontsize)
    ax.set_xlabel("Origin of Items", fontsize=label_fontsize)
    ax.set_ylabel("Probability", fontsize=label_fontsize)
    ax.set_xticks(range(5))
    ax.set_yticks([0, 0.20, 0.40])
    ax.set_xticklabels(["{}".format(location_xticklabels[idx]) for idx in range(5)], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(0, 0.45) # y轴范围

    # 绘制概率分布柱状图b1
    ax = axes[1, 0]
    ax.bar(range(n_choices), ch_scaled_dist[3], width=0.3, color='blue', alpha=0.7)
    ax.set_title(f"(b1) Output Distribution of DeepSeek-V3 w/ Chinese", fontsize=title_fontsize)
    ax.set_xlabel("Item Name", fontsize=label_fontsize)
    ax.set_ylabel("Probability", fontsize=label_fontsize)
    ax.set_xticks(range(n_choices))
    ax.set_yticks([0, 0.05, 0.10])
    ax.set_xticklabels([item.split(" - ")[0] for item in items], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(0, 0.12) # y轴范围

    # 绘制标签分布柱状图b2
    ax = axes[1, 1]
    ax.bar(range(5), ch_label_dist[3], width=0.2, color='#FEA040', alpha=0.7)
    ax.set_title("(b2) Accumulative Probability w/ Chinese", fontsize=title_fontsize)
    ax.set_xlabel("Origin of Items", fontsize=label_fontsize)
    ax.set_ylabel("Probability", fontsize=label_fontsize)
    ax.set_xticks(range(5))
    ax.set_yticks([0, 0.20, 0.40])
    ax.set_xticklabels(["{}".format(location_xticklabels[idx]) for idx in range(5)], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(0, 0.45) # y轴范围

    plt.tight_layout()
    
    # 保存图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/GGS-real_Z_Distribution-DSV3.pdf", dpi=500)

    # 显示图像
    plt.show()

    # 输出结果，以表格形式
    print("Accumulative Probability of Models:")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Probability(En|Ch) |")
    for i in range(len(model_name)):
        print("| {} | ({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}) |".format(
            model_name[i],
            en_label_dist[i][0], # 保留 4 位小数
            en_label_dist[i][1],
            en_label_dist[i][2],
            en_label_dist[i][3],
            en_label_dist[i][4],
            ch_label_dist[i][0], # 保留 4 位小数
            ch_label_dist[i][1],
            ch_label_dist[i][2], 
            ch_label_dist[i][3],
            ch_label_dist[i][4]
        ))
    print("------------------------------------------------------------------------------")

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
    scale_vactor: float= 1/5.5,
) -> None:
    """
    可视化不同模型的 Decision Log Standard Deviation (D_lstd)
    不考虑flavor和location，只考虑action num
    Task: GGS-real
    """
    # 柱状图1：横坐标是 Model Name，纵坐标是 D_lstd Value ，每个横坐标两个bar（en & ch）
    en_D_lstd = []
    ch_D_lstd = []
    en_original_dist = []
    ch_original_dist = []
    for result in evaluate_results:
        en_original_dist.append(result[0]["record"][0][0])
        ch_original_dist.append(result[1]["record"][0][0])
    
    en_scaled_dist = (np.array(en_original_dist) * scale_vactor).tolist() # scaling
    ch_scaled_dist = (np.array(ch_original_dist) * scale_vactor).tolist() # scaling

    for i in range(len(en_scaled_dist)):
        en_D_lstd.append(D_lstd(en_scaled_dist[i], vmin=vmin, vmax=vmax))
        ch_D_lstd.append(D_lstd(ch_scaled_dist[i], vmin=vmin, vmax=vmax))

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
    plt.xticks(x_model_name, model_name, fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=tick_fontsize)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.legend(loc='upper left', fontsize=legend_fontsize)

    # 指定 x 轴和 y 轴的范围
    plt.ylim(0, 0.55) # y轴范围
    plt.tight_layout()

    # 保存图片，并设置分辨率为 500 DPI
    # plt.savefig("./img/real/GGS-real_D_lstd.pdf", dpi=500)

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
    Task: GGS-real
    """
    en_original_dist = []
    ch_original_dist = []
    for result in evaluate_results:
        en_original_dist.append(result[0]["record"][0][0])
        ch_original_dist.append(result[1]["record"][0][0])
    
    en_avg_dist = (np.array(en_original_dist) * scale_vactor).tolist() # scaling
    ch_avg_dist = (np.array(ch_original_dist) * scale_vactor).tolist() # scaling

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
    plt.savefig("./img/real/GGS-real_D_cl.pdf", dpi=500)
    plt.savefig("./img/real/GGS-real_D_cl.png", dpi=500)

    # 显示图形
    plt.show()

    # 输出结果，以表格形式
    print("D_cl in GGS-real (Mean):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i],
            mean_D_cl[i] # 保留 3 位小数
        ))
    print("------------------------------------------------------------------------------")

    print("D_cl in GGS-real (Std):")
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
    该程序的用途： 可视化Task: GGS-real 中的各项指标
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
        model_path = './record/GGS_real_{}_raw-final.json'.format(m)
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
    
    # Plot 3:
    visualize_distribution(
        evaluate_results = model_data_raw, 
        model_name = model_name,
        items = model_data_raw[3][0]["selected_action_space"]["english"],
    )
    
