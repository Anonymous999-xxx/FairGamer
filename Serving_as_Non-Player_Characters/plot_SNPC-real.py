from utils import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

def update_records_with_average(total_task_info):
    for task in total_task_info:
            for i in range(len(task["record"])):
                for j in range(len(task["record"][i])):
                    inner_list = task["record"][i][j]
                    average = sum(inner_list) / len(inner_list)

                    if average >= 100.0:
                        average = 100.0
                    elif average <= 50.0:
                        average = 50.0
                    
                    task["record"][i][j] = average
    return total_task_info

def average_4th_dimension(record):
    record_array = np.array(record)
    averaged_record = np.mean(record_array, axis=-1)
    averaged_record = np.clip(averaged_record, 50, 100)
    return averaged_record.tolist()

def average_1st_dimension(record):
    record_array = np.array(record)
    averaged_record = np.mean(record_array, axis=0)
    averaged_record = np.clip(averaged_record, 50, 100)
    return averaged_record.tolist()

def update_records_through_average(total_task_info):
    for task in total_task_info:
        task["record"] = average_4th_dimension(task["record"])
        task["record"] = average_1st_dimension(task["record"])
    return total_task_info


def sort_matrix_and_labels(matrix, xticklabels, yticklabels):
    matrix = np.array(matrix)
    row_indices = np.argsort(matrix, axis=1)
    sorted_matrix = np.take_along_axis(matrix, row_indices, axis=1)
    
    col_indices = np.argsort(sorted_matrix, axis=0)
    sorted_matrix = np.take_along_axis(sorted_matrix, col_indices, axis=0)
    sorted_xticklabels = [xticklabels[i] for i in row_indices[0]]
    sorted_yticklabels = [yticklabels[i] for i in col_indices[:, 0]]
    
    return sorted_matrix, sorted_xticklabels, sorted_yticklabels

def plot_heatmap(ax, data, title, vmin, vmax, xticklabels, yticklabels, reverse=False):
    matrix = np.array(data)
    if reverse:
        matrix = matrix.T
        xticklabels, yticklabels = yticklabels, xticklabels
    
    im = ax.imshow(matrix, cmap='coolwarm', aspect=0.5, vmin=vmin, vmax=vmax)

    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 12
    num_fontsize = 10

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Career' if not reverse else 'Race', fontsize=label_fontsize)
    ax.set_ylabel('Race' if not reverse else 'Career', fontsize=label_fontsize)
    
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=tick_fontsize)

    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=num_fontsize)
    ax.margins(x=0, y=0)

    return im

def visualize_HeatMap(total_task_info, xticklabels, yticklabels):
    info = total_task_info.copy()
    game_name = info[0]['game_name'][0]

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    
    for i in range(len(info)):
        record = np.array(info[i]["record"]) / 100
        info[i]["record"] = record.tolist()

    all_data = [np.array(task["record"]) for task in info]
    vmin = min(np.min(data) for data in all_data)
    vmax = max(np.max(data) for data in all_data)
    print(f"\nvmin: {vmin}")
    print(f"\nvmax: {vmax}")
    xy_stick_lang = 'english'
    game_name = info[0]['game_name'][0]

    for i, task in enumerate(info):
        lang = task["evaluated_language"]
        lang = lang[0].upper() + lang[1:]

        if i == 0:
            title = [f"(a) {lang}"]
        else:
            title.append(f"(b) {lang}")

        record = task["record"]

        sorted_matrix, sorted_xticklabels, sorted_yticklabels = sort_matrix_and_labels(record, xticklabels[xy_stick_lang], yticklabels[xy_stick_lang])

        if i == 0:
            im = plot_heatmap(ax1, sorted_matrix, title[i], vmin, vmax, sorted_xticklabels, sorted_yticklabels, reverse=True)
        else:
            im = plot_heatmap(ax2, sorted_matrix, title[i], vmin, vmax, sorted_xticklabels, sorted_yticklabels, reverse=True)
        
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Average Value', fontsize=16)
    
    plt.subplots_adjust(left=0, right=0.9, top=0.85, bottom=0.15, wspace=0)

    plt.savefig("./img/real/HeatMap-DeepSeekV3.pdf", dpi=500)

    plt.show()

def D_lstd(data, vmin=0, vmax=1, log_bias=0.01):
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
    vmin: float=0.5,
    vmax: float=1.0,
    scale_vactor: float=1.0,
) -> None:
    en_D_lstd = []
    ch_D_lstd = []
    for result in evaluate_results:
        for i in range(len(result)):
            record = np.array(result[i]["record"]) / 100
            record = record.tolist()
            if i == 0:
                en_D_lstd.append(D_lstd(record, vmin, vmax))
            elif i == 1:
                ch_D_lstd.append(D_lstd(record, vmin, vmax))
    
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
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    x_model_name = np.arange(len(model_name)) + 1
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
    vmin: float=0.5,
    vmax: float=1.0,
    scale_vactor: float=1.0,
) -> None:
    mean_D_cl = []
    std_D_cl = []
    D_cl = []
    for result in evaluate_results:
        for i in range(len(result)):
            record = np.array(result[i]["record"]) / 100

            if np.any(record < vmin) or np.any(record > vmax):
                raise ValueError(f"must in [{vmin}, {vmax}]")

            record = (record - vmin) / (vmax - vmin)
            if i == 0:
                diff_array = record
            else:
                diff_array -= record
        diff_array = scale_vactor * np.abs(diff_array)
        D_cl.append(diff_array.flatten().tolist())
        mean_D_cl.append(np.mean(diff_array))
        std_D_cl.append(np.std(diff_array))
    
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
        (255, 127, 0),   # orange
        (106, 180, 193), # blue & green
        (112, 180, 143), # green
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

    plt.savefig("./img/real/SNPC-real_D_cl.pdf", dpi=500)
    plt.savefig("./img/real/SNPC-real_D_cl.png", dpi=500)

    plt.show()

    print("D_cl in SNPC-real (Mean):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i],
            mean_D_cl[i]
        ))
    print("------------------------------------------------------------------------------")

    print("D_cl in SNPC-real (Std):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i], 
            std_D_cl[i]
        ))
    print("------------------------------------------------------------------------------")

if __name__ == '__main__':
    game_race = read_json('./data/race-SNPC.json')
    game_race = game_race["real"]
    game_career = read_json('./data/career-SNPC.json')
    game_career = game_career["real"]
    
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
        model_path = './record/SNPC_real_{}_raw-final.json'.format(m)
        model_info = read_json(model_path)
        # print('-----------total_task_info-----------')
        # print("{} record sample: \n{}".format(m, model_info))
        # print('---------------------------------------------')
        model_data_temp = update_records_through_average(model_info)
        # print('-----------updated_total_task_info-----------')
        # print("updated {} record sample: \n{}".format(m, model_data_temp))
        # print('---------------------------------------------')
        model_data_raw.append(model_data_temp)

    # Plot 1:
    visualize_Decision_lstd(evaluate_results = model_data_raw, model_name = model_name)
    
    # Plot 2:
    visualize_D_cl(evaluate_results = model_data_raw, model_name = model_name)
    
    # Plot 3:
    visualize_HeatMap(model_data_raw[3], game_career, game_race)
