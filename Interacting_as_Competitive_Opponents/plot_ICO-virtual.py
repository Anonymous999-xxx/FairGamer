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
        for i in range(len(task["record"])): # game names
            for j in range(len(task["record"][i])): # len(versus_list)
                for k in range(len(task["record"][i][j])): # len(diplomatic_prompts)
                    for l in range(len(task["record"][i][j][k])): # len(diplomatic_options)
                        inner_list = task["record"][i][j][k][l] # repeated_num
                        total_len = len(inner_list)
                        ratio_0 = inner_list.count(0) / total_len
                        ratio_1 = inner_list.count(1) / total_len
                        ratio_2 = inner_list.count(2) / total_len
                    
                        task["record"][i][j][k][l] = [ratio_0, ratio_1, ratio_2]
        task["record"] = np.mean(task["record"], axis=3) # get average p of diplomatic_options
        task["record"] = np.mean(task["record"], axis=0) # get average p of game names
    return total_task_info

def reconstruct(raw_list: list) -> list:
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
    for i in range(len(temp_rec_list)):
        sub_list = []
        for j in range(len(temp_rec_list[i])):
            p = {}
            for k in range(len(temp_rec_list[i][j])):
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
    decisions = ["Neutral", "Oppose", "Cooperate"]
    r1, r2 = 0, 0
    
    for i in range(len(decisions)):
        for j in range(len(decisions)):
            decision1 = decisions[i]
            decision2 = decisions[j]
            payoff = payoff_matrix[decision1][decision2]
            
            r1 += p1[i] * p2[j] * payoff[0]
            r2 += p1[i] * p2[j] * payoff[1]
    
    return round(r1, 5), round(r2, 5)

def calculate_all_payoffs(data, payoff_matrix):
    for i in range(len(data)):
        record = data[i]["record"]
        payoffs = []
        for j in range(len(record)):
            country_combination = []
            for k in range(len(record[j])):
                p1 = record[j][k]["country_1-adv"]
                p2 = record[j][k]["country_2-disadv"]
                assert len(p1) == len(p2)

                r1, r2 = calculate_payoff(p1, p2, payoff_matrix) 
                country_combination.append([r1, r2])

            payoffs.append(country_combination)

        data[i]["game_matrix_reward"] = payoffs

    return data

def reward_process(data):
    reward_sum_diff = []
    reward_sum_sum = []
    reward_sum_1 = []
    reward_sum_2 = []

    for i in range(len(data)):
        game_matrix_reward = data[i]['game_matrix_reward']

        reward_sum_1_temp = []
        reward_sum_2_temp = []
    
        for j in range(len(game_matrix_reward)):
            r1 = game_matrix_reward[j][0][0]
            r2 = game_matrix_reward[j][0][1]
            reward_sum_1_temp.append(r1 + r2)

            r3 = game_matrix_reward[j][1][0]
            r4 = game_matrix_reward[j][1][1]
            reward_sum_2_temp.append(r3 + r4)
        
        reward_sum_1.append(reward_sum_1_temp)
        reward_sum_2.append(reward_sum_2_temp)

        diff_temp = np.abs(np.array(reward_sum_1_temp) - np.array(reward_sum_2_temp))
        reward_sum_diff.append(diff_temp)

        sum_temp = np.array(reward_sum_1_temp) + np.array(reward_sum_2_temp)
        reward_sum_sum.append(sum_temp)

    return reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff
    
def visualize_rewards(data):
    reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff = reward_process(data)

    x = np.arange(len(reward_sum_1[0]))
    
    width = 0.20

    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize = 8
    legend_fontsize = 10

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

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ax1.bar(x - width/2, reward_sum_1[0], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax1.bar(x + width/2, reward_sum_2[0], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax1.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax1.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax1.set_title("(a) Sum of Expected Payoff w/ English", fontsize=title_fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1.set_ylim(-3.5, 4.5)
    ax1.legend(loc='upper left', fontsize=legend_fontsize)

    ax2.bar(x - width/2, reward_sum_1[1], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax2.bar(x + width/2, reward_sum_2[1], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax2.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax2.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax2.set_title("(b) Sum of Expected Payoff w/ Chinese", fontsize=title_fontsize)
    ax2.set_xticks(x)
    ax2.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    ax2.set_ylim(-3.5, 4.5)
    ax2.legend(loc='lower right', fontsize=legend_fontsize)

    plt.tight_layout()

    plt.savefig("./img/virtual/ICO-virtual_Payoffs-DSV3.pdf", dpi=500)

    plt.show()

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 7))

    ax3.bar(x, 0.5*np.array(reward_sum_sum[0]), width, color='teal')
    ax3.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax3.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax3.set_title("(a) Avg Sum of Expected Payoff w/ English", fontsize=title_fontsize)
    ax3.set_xticks(x)
    ax3.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax3.tick_params(axis='y', labelsize=tick_fontsize)
    ax3.set_ylim(-3.5, 4.5)
    
    ax4.bar(x, 0.5*np.array(reward_sum_sum[1]), width, color='teal')
    ax4.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax4.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax4.set_title("(b) Avg Sum of Expected Payoff w/ Chinese", fontsize=title_fontsize)
    ax4.set_xticks(x)
    ax4.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax4.tick_params(axis='y', labelsize=tick_fontsize)
    ax4.set_ylim(-3.5, 4.5)

    plt.tight_layout()

    plt.savefig("./img/virtual/ICO-virtual_Avg_Payoffs-DSV3.pdf", dpi=500)

    plt.show()

    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 7))

    ax5.bar(x - width/2, reward_sum_1[0], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax5.bar(x + width/2, reward_sum_2[0], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax5.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax5.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax5.set_title("(a) English", fontsize=title_fontsize)
    ax5.set_xticks(x)
    ax5.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax5.tick_params(axis='y', labelsize=tick_fontsize)
    ax5.set_ylim(-3.5, 4.5)
    ax5.axhline(y=0.12, color='darkgreen', linestyle='--', linewidth=2)
    ax5.text(x[-19]+1, 0.12 + 0.1, 'Unbiased Payoff of LLM', color='darkgreen', fontsize=label_fontsize)
    ax5.axhline(y=4.0, color='black', linestyle='-.', linewidth=2)
    ax5.text(x[-13]+1, 4.0 + 0.1, 'Pareto Optimal', color='black', fontsize=label_fontsize)
    ax5.legend(loc='upper left', fontsize=legend_fontsize)

    ax6.bar(x - width/2, reward_sum_1[1], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax6.bar(x + width/2, reward_sum_2[1], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax6.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax6.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax6.set_title("(b) Chinese", fontsize=title_fontsize)
    ax6.set_xticks(x)
    ax6.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax6.tick_params(axis='y', labelsize=tick_fontsize)
    ax6.set_ylim(-3.5, 4.5)
    ax6.axhline(y=3.08, color='darkgreen', linestyle='--', linewidth=2)
    ax6.text(x[-19]+1, 3.08 + 0.1, 'Unbiased Payoff of LLM', color='darkgreen', fontsize=label_fontsize)
    ax6.axhline(y=4.0, color='black', linestyle='-.', linewidth=2)
    ax6.text(x[-13]+1, 4.0 + 0.1, 'Pareto Optimal', color='black', fontsize=label_fontsize)
    ax6.legend(loc='lower right', fontsize=legend_fontsize)

    plt.tight_layout()
    
    plt.savefig("./img/virtual/ICO-virtual_Payoffs-Comparison.pdf", dpi=500)
    
    plt.show()

    fig4, (ax7) = plt.subplots(1, 1, figsize=(7, 7))

    title_fontsize = 16
    label_fontsize = 16
    tick_fontsize = 8
    legend_fontsize = 10

    ax7.bar(x - width/2, reward_sum_1[0], width, label="Advantaged Country i vs Disadvantaged Country j", color='royalblue')
    ax7.bar(x + width/2, reward_sum_2[0], width, label="Disadvantaged Country i vs Advantaged Country j", color='lightcoral')
    ax7.set_xlabel("Two Sides in the Game", fontsize=label_fontsize)
    ax7.set_ylabel("Expected Payoff Value", fontsize=label_fontsize)
    ax7.set_xticks(x)
    ax7.set_xticklabels(versus_labels["english"], rotation=90, rotation_mode='anchor', ha='right', fontsize=tick_fontsize)
    ax7.tick_params(axis='y', labelsize=tick_fontsize)
    ax7.legend(loc='upper left', fontsize=legend_fontsize)
    ax7.set_ylim(-3.5, 4.5)

    ax7.axhline(y=0.12, color='darkgreen', linestyle='--', linewidth=2)
    ax7.text(x[-19]+1, 0.12 + 0.1, 'Unbiased Payoff of LLM', color='darkgreen', fontsize=14)
    ax7.axhline(y=4.0, color='black', linestyle='-.', linewidth=2)
    ax7.text(x[-13]+1, 4.0 + 0.1, 'Pareto Optimal', color='black', fontsize=14)
    
    plt.tight_layout()
    
    plt.savefig("./img/virtual/ICO-virtual_Payoffs-Comparison-En.pdf", dpi=500)
    
    plt.show()

def D_lstd(data, vmin=0, vmax=1, log_bias=0.01):
    """
    Computation of the Decision Log Standard Deviation (D_lstd)
    :param data: 2d list (2d distribution)
    :return: D_lstd
    """
    data = np.array(data)
    if vmin != None and vmax != None:
        if np.any(data < vmin) or np.any(data > vmax):
            print(data)
            raise ValueError(f"must in [{vmin}, {vmax}]")

        data = (data - vmin) / (vmax - vmin)

    data = data.flatten()
    data = data + log_bias
    return np.std(np.log(data))

def visualize_Decision_lstd(
    evaluate_results: list, 
    model_name: list,
    min_reward_sum: float=-4.0,
    max_reward_sum: float=4.0,
    scale_vactor: float=1.0,
) -> None:
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
    
    title_fontsize = 16
    label_fontsize = 28
    tick_fontsize = 20
    legend_fontsize = 24
    colors_255 = [
        (88, 97, 172),   # blue
        (255, 127, 0),   # orange
        (106, 180, 193), # blue & green
        (112, 180, 143), # green
        (107, 126, 185), # blue
        (254, 160, 64),  # orange
        (106, 184, 103)  # green
    ]

    colors = [(r/255, g/255, b/255) for r, g, b in colors_255]

    plt.figure(figsize=(10, 6))
    x_model_name = np.arange(len(model_name))
    width = 0.35

    plt.bar(
        x_model_name - width/2, 
        (np.array(en_D_lstd)*scale_vactor).tolist(), 
        width=width, 
        color=colors[4], 
        alpha=0.7, 
        label='English'
    )
    plt.bar(
        x_model_name + width/2, 
        (np.array(ch_D_lstd)*scale_vactor).tolist(), 
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

    plt.ylim(0, 0.55)
    plt.tight_layout()

    plt.show()

    print("D_lstd of Models:")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | ({:.3f}) ({:.3f}) |".format(
            model_name[i],
            en_D_lstd[i],
            ch_D_lstd[i]
        ))
    print("------------------------------------------------------------------------------")

def visualize_D_cl(
    evaluate_results: list, 
    model_name: list,
    min_reward_sum: float=-4.0,
    max_reward_sum: float=4.0,
    scale_vactor: float=1.0,
) -> None:
    mean_D_cl = []
    std_D_cl = []
    D_cl = [] # 2d D_cl
    for result in evaluate_results:
        reward_sum_1, reward_sum_2, reward_sum_sum, reward_sum_diff = reward_process(result)
        for i in range(len(reward_sum_sum)):
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

    title_fontsize = 16
    label_fontsize = 30
    tick_fontsize = 24
    legend_fontsize = 24

    plt.figure(figsize=(12, 7))
    box = plt.boxplot(
        D_cl, 
        patch_artist=True,
        flierprops=dict(marker='None')
    )
    
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
    colors = [(r/255, g/255, b/255) for r, g, b in colors_255]

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=1)

    for median in box['medians']:
        median.set(color='black', linewidth=1)

    plt.xlabel('Model', fontsize=label_fontsize)
    plt.ylabel('Difference Value', fontsize=label_fontsize)
    plt.xticks(x_model_name, model_name, fontsize=16, rotation=45, ha='right')
    plt.yticks([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60], fontsize=tick_fontsize)
    plt.ylim(-0.05, 0.65)
    plt.tight_layout()

    plt.savefig("./img/virtual/ICO-virtual_D_cl.pdf", dpi=500)
    plt.savefig("./img/virtual/ICO-virtual_D_cl.png", dpi=500)

    plt.show()

    print("D_cl in ICO-virtual (Mean):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i],
            mean_D_cl[i]
        ))
    print("------------------------------------------------------------------------------")

    print("D_cl in ICO-virtual (Std):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i], 
            std_D_cl[i]
        ))
    print("------------------------------------------------------------------------------")

if __name__ == '__main__':
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
        model_path = './record/ICO_virtual_{}_raw-final.json'.format(m)
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
