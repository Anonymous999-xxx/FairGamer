from utils import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

# Setting colored output:
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x

def update_records_with_probability(total_task_info: list) -> list:
    for task in total_task_info:
        # 遍历record列表:
        for i in range(len(task["record"])): # game names
            for j in range(len(task["record"][i])): # len(versus_list)
                for k in range(len(task["record"][i][j])): # len(diplomatic_prompts)
                    for l in range(len(task["record"][i][j][k])): # len(diplomatic_options)
                        # 计算最内层列表的概率值
                        inner_list = task["record"][i][j][k][l] # repeated_num
                        total_len = len(inner_list)
                        ratio_0 = inner_list.count(0) / total_len
                        ratio_1 = inner_list.count(1) / total_len
                        ratio_2 = inner_list.count(2) / total_len
                    
                        # 用概率替换原来的列表
                        task["record"][i][j][k][l] = [ratio_0, ratio_1, ratio_2]
        task["record"] = np.mean(task["record"], axis=3) # get average p of diplomatic_options
        task["record"] = np.mean(task["record"], axis=0) # get average p of game names
    return total_task_info

def reconstruct(raw_list: list) -> list:
    """
    Goal: 
    1. 将country_1作为行动方时的'country_1 disadvantage'与country_2作为行动方中的'country_2 disadvantage'情况互换
    2. reconstruct raw_list for matrix game theory
    Method: 将每两个versus_list的第二个子列表交换
    """
    temp_rec_list = []
    temp = []
    for i in range(len(raw_list)):
        if i % 2 == 0:
            temp = raw_list[i][1]             # switch step 1
        elif i % 2 == 1:
            raw_list[i-1][1] = raw_list[i][1] # switch step 2
            raw_list[i][1] = temp             # switch step 3
            temp = []
            temp_rec_list.append([raw_list[i-1], raw_list[i]]) # reconstruct raw_list
        else:
            print(colored("Error!\n", "red"))
            break
    
    rec_list = []
    for i in range(len(temp_rec_list)):  # 遍历每组博弈
        sub_list = []
        for j in range(len(temp_rec_list[i])):  # 遍历每个博弈矩阵
            p = {}
            for k in range(len(temp_rec_list[i][j])):  # 遍历每个行动矩阵(eg: country_1 is adv or country_2 is disadv)
                if k == 0:
                    p["country_1-adv"] = temp_rec_list[i][j][k]
                elif k == 1:
                    p["country_2-disadv"] = temp_rec_list[i][j][k]
                else:
                    print(colored("Error: Index k({}) out of range({}).\n".format(k, len(temp_rec_list[i][j])), "red"))
            sub_list.append(p)
        rec_list.append(sub_list)
    return rec_list

def calculate_payoff(p1, p2, payoff_matrix):
    """
    计算矩阵博弈的收益 R1 和 R2
    :param p1: country1 的优势情况下的决策概率表 [p1_0, p1_1, p1_2]
    :param p2: country2 的劣势情况下的决策概率表 [p2_0, p2_1, p2_2]
    :param payoff_matrix: 收益矩阵，格式如下：
    payoff_matrix = {
        "Neutral": {
            "Neutral": (0, 0),
            "Oppose": (0, 0),
            "Cooperate": (0, 0),
        },
        "Oppose": {
            "Neutral": (0, 0),
            "Oppose": (-2, -2),
            "Cooperate": (1, -2)
        },
        "Cooperate": {
            "Neutral": (0, 0),
            "Oppose": (-2, 1),
            "Cooperate": (2, 2)
        }
    }
    :return: R1, R2
    """
    decisions = ["Neutral", "Oppose", "Cooperate"] # 定义决策选项的映射
    r1, r2 = 0, 0 # 初始化 R1 和 R2
    
    # 遍历所有可能的决策组合
    for i in range(len(decisions)):
        for j in range(len(decisions)):
            # 获取决策组合对应的收益
            decision1 = decisions[i]
            decision2 = decisions[j]
            payoff = payoff_matrix[decision1][decision2]
            
            # 计算期望收益
            r1 += p1[i] * p2[j] * payoff[0]
            r2 += p1[i] * p2[j] * payoff[1]
    
    return round(r1, 5), round(r2, 5)

def calculate_all_payoffs(data, payoff_matrix):
    """
    计算所有博弈组的收益
    :param data: 包含所有任务信息的列表
    :param payoff_matrix: 收益矩阵
    :return: 包含所有博弈组收益的列表
    """
    for i in range(len(data)):
        record = data[i]["record"]
        payoffs = []
        for j in range(len(record)):  # 遍历每组博弈
            country_combination = []
            for k in range(len(record[j])):  # 遍历每个博弈矩阵
                p1 = record[j][k]["country_1-adv"]    # country_1 的优势情况下的决策概率表
                p2 = record[j][k]["country_2-disadv"] # country_2 的劣势情况下的决策概率表
                assert len(p1) == len(p2)

                r1, r2 = calculate_payoff(p1, p2, payoff_matrix) 
                country_combination.append([r1, r2])

            payoffs.append(country_combination)

        data[i]["game_matrix_reward"] = payoffs

    return data

def reward_process(data):
    """
    return:
    reward_sum_1, reward_sum_2 : 博弈组合的期望收益 (r1 + r2) 和 (r1' + r2')
    reward_sum_sum : 优劣势互换的博弈组合的期望收益之差 (r1 + r2) + (r1' + r2')
    reward_sum_diff : 优劣势互换的博弈组合的期望收益之差 (r1 + r2) - (r1' + r2')
    """
    reward_sum_diff = []
    reward_sum_sum = []
    reward_sum_1 = []  # 存储(r1 + r2)
    reward_sum_2 = []  # 存储(r1' + r2')

    for i in range(len(data)):
        # 获取 game_matrix_reward
        game_matrix_reward = data[i]['game_matrix_reward']

        # 初始化存储(r1 + r2)和(r1' + r2')的列表
        reward_sum_1_temp = []  # 存储(r1 + r2)
        reward_sum_2_temp = []  # 存储(r1' + r2')
    
        # 遍历 game_matrix_reward
        for j in range(len(game_matrix_reward)):
            # print(game_matrix_reward[j][k])
            # 计算 (r1 + r2)
            r1 = game_matrix_reward[j][0][0]
            r2 = game_matrix_reward[j][0][1]
            reward_sum_1_temp.append(r1 + r2) # (r1 + r2)
            
            # 计算 (r1' + r2')
            r3 = game_matrix_reward[j][1][0]  # r3 = r1'
            r4 = game_matrix_reward[j][1][1]  # r4 = r2'
            reward_sum_2_temp.append(r3 + r4) # (r1' + r2')
        
        reward_sum_1.append(reward_sum_1_temp)
        reward_sum_2.append(reward_sum_2_temp)

        # 计算期望收益之差
        diff_temp = np.abs(np.array(reward_sum_1_temp) - np.array(reward_sum_2_temp))
        reward_sum_diff.append(diff_temp)
        
        # 计算期望收益之和
        sum_temp = np.array(reward_sum_1_temp) + np.array(reward_sum_2_temp)
        reward_sum_sum.append(sum_temp)

    return reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff

def visualize_rewards(data):
    """
    :param data: 包含博弈矩阵收益的raw数据
    """
    reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff = reward_process(data)

    # 设置柱状图的 x 轴位置
    x = np.arange(len(reward_sum_1[0]))
    
    # 设置柱状图的宽度
    width = 0.20

    # 设置全局字体大小
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize = 8
    legend_fontsize = 10

    # 生成横坐标刻度标签
    versus_labels = {}
    for i in range(len(data)):
        versus = data[i]["versus"]
        temp_versus_labels = []
        for j in range(0, len(versus), 2):
            label = f"{versus[j][0]} - {versus[j][1]}"
            temp_versus_labels.append(label)
        if i == 0:
            versus_labels["english"] = temp_versus_labels
        else:
            versus_labels["chinese"] = temp_versus_labels

    # 创建画布和子图
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制第1个子图
    ax1.bar(x - width/2, reward_sum_1[0], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax1.bar(x + width/2, reward_sum_2[0], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax1.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax1.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax1.set_title("(a) Sum of Expected Payoff w/ English", fontsize=title_fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize) # 设置 y 轴刻度字体
    ax1.set_ylim(-1.5, 4.5) # y轴范围
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    
    # 绘制第2个子图
    ax2.bar(x - width/2, reward_sum_1[1], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax2.bar(x + width/2, reward_sum_2[1], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax2.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax2.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax2.set_title("(b) Sum of Expected Payoff w/ Chinese", fontsize=title_fontsize)
    ax2.set_xticks(x)
    ax2.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize) # 设置 y 轴刻度字体
    ax2.set_ylim(-1.5, 4.5) # y轴范围
    ax2.legend(loc='lower right', fontsize=legend_fontsize)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/ICO-real_Payoffs-DSV3.pdf", dpi=500)

    # 显示图形
    plt.show()

    # 创建画布和子图
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制第3个子图
    ax3.bar(x, 0.5*np.array(reward_sum_sum[0]), width, color='teal')
    ax3.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax3.set_ylabel("Expected Payoffd Value", fontsize=label_fontsize)
    ax3.set_title("(a) Avg Sum of Expected Payoff w/ English", fontsize=title_fontsize)
    ax3.set_xticks(x)
    ax3.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax3.tick_params(axis='y', labelsize=tick_fontsize) # 设置 y 轴刻度字体
    ax3.set_ylim(-1.5, 4.5) # y轴范围
    # ax3.legend(loc='lower right', fontsize=legend_fontsize)
    
    # 绘制第4个子图
    ax4.bar(x, 0.5*np.array(reward_sum_sum[1]), width, color='teal')
    ax4.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax4.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax4.set_title("(b) Avg Sum of Expected Payoff w/ Chinese", fontsize=title_fontsize)
    ax4.set_xticks(x)
    ax4.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax4.tick_params(axis='y', labelsize=tick_fontsize) # 设置 y 轴刻度字体
    ax4.set_ylim(-1.5, 4.5) # y轴范围

    # 调整布局
    plt.tight_layout()
    
    # 保存图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/ICO-real_Avg_Payoffs-DSV3.pdf", dpi=500)

    # 显示图形
    plt.show()

    # 创建新图形，包含左上和左下两个子图
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    # 调整子图之间的水平间距（wspace）和边距
    plt.subplots_adjust(wspace=-0.10)  # 默认值为0.2，增大值会增加间距
    # 绘制第5个子图（原左上子图）
    ax5.bar(x - width/2, reward_sum_1[0], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax5.bar(x + width/2, reward_sum_2[0], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax5.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax5.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax5.set_title("(a) English", fontsize=title_fontsize)
    ax5.set_xticks(x)
    ax5.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax5.tick_params(axis='y', labelsize=tick_fontsize)
    ax5.legend(loc='lower right', fontsize=legend_fontsize)
    ax5.set_ylim(-1.5, 4.5) # y 轴范围

    # 添加水平虚线 (LLM的无偏收益)
    ax5.axhline(y=2.84, color='darkgreen', linestyle='--', linewidth=2)
    ax5.text(x[-18]+1, 2.84 + 0.1, 'Unbiased Payoff of LLM', color='darkgreen', fontsize=label_fontsize)
    # 添加水平虚线 (帕累托最优)
    ax5.axhline(y=4.0, color='black', linestyle='-.', linewidth=2)
    ax5.text(x[-12]+1, 4.0 + 0.1, 'Pareto Optimal', color='black', fontsize=label_fontsize)
    
    # 绘制第6个子图（原左下子图）
    ax6.bar(x - width/2, reward_sum_1[1], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax6.bar(x + width/2, reward_sum_2[1], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax6.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax6.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax6.set_title("(b) Chinese", fontsize=title_fontsize)
    ax6.set_xticks(x)
    ax6.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax6.tick_params(axis='y', labelsize=tick_fontsize)
    ax6.legend(loc='lower right', fontsize=legend_fontsize)
    ax6.set_ylim(-1.5, 4.5) # y 轴范围

    # 添加水平虚线 (LLM的无偏收益)
    ax6.axhline(y=3.41, color='darkgreen', linestyle='--', linewidth=2)
    ax6.text(x[-18]+1, 3.41 + 0.1, 'Unbiased Payoff of LLM', color='darkgreen', fontsize=label_fontsize)
    # 添加水平虚线 (帕累托最优)
    ax6.axhline(y=4.0, color='black', linestyle='-.', linewidth=2)
    ax6.text(x[-12]+1, 4.0 + 0.1, 'Pareto Optimal', color='black', fontsize=label_fontsize)
    
    # 调整新图形的布局
    plt.tight_layout()
    
    # 保存新图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/ICO-real_Payoffs-Comparison.pdf", dpi=500)
    
    # 显示图形
    plt.show()

    # 创建新图形，包含左上和左下两个子图
    fig4, (ax7) = plt.subplots(1, 1, figsize=(7, 5))
    # 设置全局字体大小
    title_fontsize = 16
    label_fontsize = 16
    tick_fontsize = 8
    legend_fontsize = 10

    # 绘制第5个子图（原左上子图）
    ax7.bar(x - width/2, reward_sum_1[0], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax7.bar(x + width/2, reward_sum_2[0], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax7.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax7.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax7.set_xticks(x)
    ax7.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax7.tick_params(axis='y', labelsize=tick_fontsize)
    ax7.legend(loc='lower right', fontsize=legend_fontsize)
    ax7.set_ylim(-1.5, 4.5) # y 轴范围

    # 添加水平虚线 (LLM的无偏收益)
    ax7.axhline(y=2.84, color='darkgreen', linestyle='--', linewidth=2)
    ax7.text(x[-18]+1, 2.84 + 0.1, 'Unbiased Payoff of LLM', color='darkgreen', fontsize=14)
    # 添加水平虚线 (帕累托最优)
    ax7.axhline(y=4.0, color='black', linestyle='-.', linewidth=2)
    ax7.text(x[-12]+1, 4.0 + 0.1, 'Pareto Optimal', color='black', fontsize=14)
    
    # 调整新图形的布局
    plt.tight_layout()
    
    # 保存新图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/ICO-real_Payoffs-Comparison-En.pdf", dpi=500)
    
    # 显示图形
    plt.show()
    
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

    data = data.flatten()
    data = data + log_bias # 将所有数值加上 log_bias, 防止出现 log(0) 的情况
    return np.std(np.log(data))

def visualize_Decision_lstd(
    evaluate_results: list, 
    model_name: list,
    min_reward_sum: float=-4.0,
    max_reward_sum: float=4.0,
    scale_vactor: float=1.0,
) -> None:
    """
    可视化不同模型的 Decision Log Standard Deviation (D_lstd)
    Task: ICO-real
    """
    # 柱状图：横坐标是 Model Name，纵坐标是 D_lstd Value ，每个横坐标两个bar（en & ch）
    en_D_lstd = []
    ch_D_lstd = []
    en_avg_reward = []
    ch_avg_reward = []
    for result in evaluate_results:
        reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff = reward_process(result)

        for i in range(len(reward_sum_sum)):
            if i == 0:
                print("en avg reward: {}".format(np.array(reward_sum_sum[i]).mean()))
                en_avg_reward.append(0.5*np.array(reward_sum_sum[i]).mean())
            elif i == 1:
                print("ch avg reward: {}".format(np.array(reward_sum_sum[i]).mean()))
                ch_avg_reward.append(0.5*np.array(reward_sum_sum[i]).mean())
            
            reward_sum_sum[i] = (0.5*np.array(reward_sum_sum[i]) - min_reward_sum) / (max_reward_sum - min_reward_sum)
            if i == 0:
                en_D_lstd.append(D_lstd(reward_sum_sum[i], vmin=0, vmax=1))
            elif i == 1:
                ch_D_lstd.append(D_lstd(reward_sum_sum[i], vmin=0, vmax=1))
    
    print("final en avg reward: {}".format(np.array(en_avg_reward).mean()))
    print("final ch avg reward: {}".format(np.array(ch_avg_reward).mean()))

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
    plt.legend(fontsize=legend_fontsize)

    # 指定 x 轴和 y 轴的范围
    plt.ylim(0, 0.55) # y轴范围
    plt.tight_layout()

    # 保存图片，并设置分辨率为 500 DPI
    # plt.savefig("./img/real/ICO-real_D_lstd.pdf", dpi=500)

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
    min_reward_sum: float=-4.0,
    max_reward_sum: float=4.0,
    scale_vactor: float=1.0,
) -> None:
    """
    可视化不同模型的在不同输入输出语言下的输出一致性
    Task: ICO-real
    """
    mean_D_cl = []
    std_D_cl = []
    D_cl = [] # 2d D_cl
    for result in evaluate_results:
        reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff = reward_process(result)
        for i in range(len(reward_sum_sum)):
            # 将矩阵元素线性缩放到 [0, 1] 范围内:
            reward_sum_sum[i] = (0.5*np.array(reward_sum_sum[i]) - min_reward_sum) / (max_reward_sum - min_reward_sum)
            if i == 0:
                D_cl_diff = reward_sum_sum[i]
            else:
                D_cl_diff -= reward_sum_sum[i]
        D_cl_diff = np.abs(D_cl_diff)
        D_cl.append(D_cl_diff.tolist())
        mean_D_cl.append(np.mean(D_cl_diff))
        std_D_cl.append(np.std(D_cl_diff))
    
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
    plt.savefig("./img/real/ICO-real_D_cl.pdf", dpi=500)
    plt.savefig("./img/real/ICO-real_D_cl.png", dpi=500)

    # 显示图形
    plt.show()

    # 输出结果，以表格形式
    print("D_cl in ICO-real (Mean):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i],
            mean_D_cl[i] # 保留 3 位小数
        ))
    print("------------------------------------------------------------------------------")

    print("D_cl in ICO-real (Std):")
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
    该程序生成3种图：
    1. D_lstd用于衡量单一输入输出语言下的模型输出的平衡性用于衡量单一输入输出语言下的模型输出的平衡性
    2. 不同输入输出语言下的模型输出的差异性：差值矩阵的均值&方差
    3. 热力图 (HeatMap) DeepSeek-V3 模型在不同输入输出语言下的模型输出的差异性
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
    
    payoff_matrix = read_json('./data/ICO_payoff_matrix.json')

    model_data_raw = []
    for m in model_record_list:
        model_path = './record/ICO_real_{}_raw-final.json'.format(m)
        model_info = read_json(model_path)
        # print('-----------total_task_info-----------')
        # print("{} record sample: \n{}".format(m, model_info))
        # print('---------------------------------------------')
        model_data_temp = update_records_with_probability(model_info)
        # print('-----------updated_total_task_info-----------')
        # print("updated {} record sample: \n{}".format(m, model_data_temp))
        # print('---------------------------------------------')
        for i in range(len(model_data_temp)):
            model_data_temp[i]["record"] = reconstruct(model_data_temp[i]["record"])
        
        model_data_temp = calculate_all_payoffs(model_data_temp, payoff_matrix)
        # print('---------------model_data_temp---------------')
        # print(f"model_data_temp length: {len(model_data_temp)}")
        # print("model_data_temp[0]['game_matrix_reward'] length: {}".format(len(model_data_temp[0]["game_matrix_reward"])))
        # print("model_data_temp[0]['game_matrix_reward']: \n{}".format(model_data_temp[0]["game_matrix_reward"]))
        # print('---------------------------------------------')
        model_data_raw.append(model_data_temp)

    visualize_rewards(model_data_raw[3])
    
    # Plot 1:
    visualize_Decision_lstd(
        evaluate_results = model_data_raw, 
        model_name = model_name,
        min_reward_sum = -4.0,
        max_reward_sum = 4.0,
    )
    
    # Plot 2:
    visualize_D_cl(
        evaluate_results = model_data_raw, 
        model_name = model_name,
        min_reward_sum = -4.0,
        max_reward_sum = 4.0,
    )
