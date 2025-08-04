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
        #if task.get("eval_selection") == "race":
            # 遍历record列表
            for i in range(len(task["record"])):
                for j in range(len(task["record"][i])):
                    # 计算最内层列表的平均值
                    inner_list = task["record"][i][j]
                    average = sum(inner_list) / len(inner_list)

                    if average >= 100.0:
                        average = 100.0
                    elif average <= 50.0:
                        average = 50.0
                    
                    # 用平均值替换原来的列表
                    task["record"][i][j] = average
    return total_task_info

def average_4th_dimension(record):
    """
    对第四维（最内层）的列表求平均
    :param record: 4维列表
    :return: 3维列表，第四维被平均值替换
    """
    record_array = np.array(record)
    averaged_record = np.mean(record_array, axis=-1)  # 对第四维求平均
    averaged_record = np.clip(averaged_record, 50, 100)  # 限制范围
    return averaged_record.tolist()

def average_1st_dimension(record):
    """
    对第一维（最外层）的列表求平均
    :param record: 3维列表（第四维已经被平均）
    :return: 2维列表，第一维被平均值替换
    """
    record_array = np.array(record)
    averaged_record = np.mean(record_array, axis=0)  # 对第一维求平均
    averaged_record = np.clip(averaged_record, 50, 100)  # 限制范围
    return averaged_record.tolist()

def update_records_through_average(total_task_info):
    """
    对每个任务的record进行处理：
    1. 对第四维（最内层）求平均
    2. 对第一维（最外层）求平均
    :param total_task_info: 包含任务信息的列表
    :return: 处理后的任务信息
    """
    for task in total_task_info:
        # 对第四维求平均
        task["record"] = average_4th_dimension(task["record"])
        # 对第一维求平均
        task["record"] = average_1st_dimension(task["record"])
    return total_task_info


def sort_matrix_and_labels(matrix, xticklabels, yticklabels):
    """
    对矩阵进行行重排和列重排，并调整横坐标和纵坐标标签
    :param matrix: 二维矩阵数据
    :param xticklabels: 横坐标标签列表
    :param yticklabels: 纵坐标标签列表
    :return: 排序后的矩阵、横坐标标签、纵坐标标签
    """
    # 将矩阵转换为 NumPy 数组
    matrix = np.array(matrix)
    
    # 对矩阵的每一行进行排序，并记录排序后的索引
    row_indices = np.argsort(matrix, axis=1)
    sorted_matrix = np.take_along_axis(matrix, row_indices, axis=1)
    
    # 对排序后的矩阵的每一列进行排序，并记录排序后的索引
    col_indices = np.argsort(sorted_matrix, axis=0)
    sorted_matrix = np.take_along_axis(sorted_matrix, col_indices, axis=0)
    
    # 调整横坐标和纵坐标标签
    # 行重排后的横坐标标签
    sorted_xticklabels = [xticklabels[i] for i in row_indices[0]]  # 使用第一行的排序结果
    # 列重排后的纵坐标标签
    sorted_yticklabels = [yticklabels[i] for i in col_indices[:, 0]]  # 使用第一列的排序结果
    
    return sorted_matrix, sorted_xticklabels, sorted_yticklabels

def plot_heatmap(ax, data, title, vmin, vmax, xticklabels, yticklabels, reverse=False):
    """
    在指定的子图上绘制二维矩阵的热力图
    :param ax: 子图对象
    :param data: 二维矩阵数据
    :param title: 图的标题
    :param vmin: 颜色映射的最小值
    :param vmax: 颜色映射的最大值
    :param xticklabels: 横坐标标签列表
    :param yticklabels: 纵坐标标签列表
    :param reverse: 是否互换xy轴
    """
    # 将数据转换为 NumPy 数组
    matrix = np.array(data)
    
    # 如果 reverse 为 True，则互换 xy 轴
    if reverse:
        matrix = matrix.T  # 转置矩阵
        xticklabels, yticklabels = yticklabels, xticklabels  # 交换标签
    
    # 在子图上绘制热力图，并指定 vmin 和 vmax
    im = ax.imshow(matrix, cmap='coolwarm', aspect=0.5, vmin=vmin, vmax=vmax)

    # 设置全局字体大小
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 12
    num_fontsize = 10

    # 设置标题和坐标轴标签
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Career' if not reverse else 'Race', fontsize=label_fontsize)  # 根据 reverse 调整标签
    ax.set_ylabel('Race' if not reverse else 'Career', fontsize=label_fontsize)  # 根据 reverse 调整标签
    
    # 设置自定义的横坐标和纵坐标标签
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=tick_fontsize)  # 旋转标签以避免重叠
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=tick_fontsize)

    # 在每个格子上显示数值
    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',  # 显示两位小数
                          ha="center", va="center", color="black", fontsize=num_fontsize)
    
    # 减少热力图周围的边距
    ax.margins(x=0, y=0)

    return im

def visualize_HeatMap(total_task_info, xticklabels, yticklabels):
    """
    遍历 total_task_info，将每个字典的二维矩阵热力图绘制在一张图中的子图上，并共享一个颜色条
    :param total_task_info: 任务信息列表
    :param xticklabels: 多语言的横坐标标签列表
    :param yticklabels: 多语言的纵坐标标签列表
    """
    info = total_task_info.copy()
    game_name = info[0]['game_name'][0]

    # 创建一个大图，使用 GridSpec 控制布局
    fig = plt.figure(figsize=(14, 7))  # 调整画布大小
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0)  # 三个子图：热力图1、热力图2、颜色条
    # title_old = f'The Game "{game_name}" in Task DSR (DeepSeek-V3)'
    # title_new = title_old.replace(game_name, r'$\mathit{' + game_name + '}$')
    # fig.suptitle(title_new, fontsize=16)
    
    # 创建子图
    ax1 = fig.add_subplot(gs[0])  # 第一个子图：热力图1
    ax2 = fig.add_subplot(gs[1])  # 第二个子图：热力图2
    cax = fig.add_subplot(gs[2])  # 第三个子图：颜色条
    
    # 对数据进行归一化处理
    for i in range(len(info)):
        record = np.array(info[i]["record"]) / 100
        info[i]["record"] = record.tolist()

    # 找到所有数据的最小值和最大值，用于统一颜色映射范围
    all_data = [np.array(task["record"]) for task in info]
    vmin = min(np.min(data) for data in all_data)  # 全局最小值
    vmax = max(np.max(data) for data in all_data)  # 全局最大值
    print(f"\nvmin: {vmin}")
    print(f"\nvmax: {vmax}")

    # 遍历 total_task_info 并绘制每个字典的热力图
    xy_stick_lang = 'english'
    game_name = info[0]['game_name'][0]

    for i, task in enumerate(info):
        lang = task["evaluated_language"]
        lang = lang[0].upper() + lang[1:]
        # 生成标题
        if i == 0:
            title = [f"(a) {lang}"]
        else:
            title.append(f"(b) {lang}")
        # 提取二维矩阵
        record = task["record"]
        # 对矩阵进行排序，并调整横坐标和纵坐标标签
        sorted_matrix, sorted_xticklabels, sorted_yticklabels = sort_matrix_and_labels(record, xticklabels[xy_stick_lang], yticklabels[xy_stick_lang])
        # 在对应的子图上绘制热力图
        if i == 0:
            im = plot_heatmap(ax1, sorted_matrix, title[i], vmin, vmax, sorted_xticklabels, sorted_yticklabels, reverse=True)
        else:
            im = plot_heatmap(ax2, sorted_matrix, title[i], vmin, vmax, sorted_xticklabels, sorted_yticklabels, reverse=True)
        
    # 在第三个子图中添加颜色条
    cbar = fig.colorbar(im, cax=cax)  # 先创建 colorbar
    cbar.set_label('Average Value', fontsize=16)  # 再单独设置字体大小
    
    # 调整整体布局
    plt.subplots_adjust(left=0, right=0.9, top=0.85, bottom=0.15, wspace=0)  # 调整左右边距，使整体布局居中

    # 保存图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/HeatMap-DeepSeekV3.pdf", dpi=500)

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
    vmin: float=0.5,
    vmax: float=1.0,
    scale_vactor: float=1.0,
) -> None:
    """
    可视化不同模型的 Decision Log Standard Deviation (D_lstd)
    Task: SNPC-real
    """
    # 柱状图：横坐标是 Model Name，纵坐标是 D_lstd Value ，每个横坐标两个bar（en & ch）
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

    # 指定 x 轴和 y 轴的范围
    plt.ylim(0, 0.55) # y轴范围
    plt.tight_layout()

    # 保存图片，并设置分辨率为 500 DPI
    # plt.savefig("./img/real/SNPC-real_D_lstd.pdf", dpi=500)

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
    vmin: float=0.5,
    vmax: float=1.0,
    scale_vactor: float=1.0,
) -> None:
    """
    可视化不同模型的在不同输入输出语言下的输出一致性
    Task: SNPC-real
    """
    mean_D_cl = []
    std_D_cl = []
    D_cl = [] # flatten 2d D_cl
    for result in evaluate_results:
        for i in range(len(result)):
            record = np.array(result[i]["record"]) / 100

            # 检查矩阵元素是否在 [vmin, vmax] 范围内
            if np.any(record < vmin) or np.any(record > vmax):
                raise ValueError(f"矩阵元素必须在 [{vmin}, {vmax}] 范围内")

            # 将矩阵元素线性缩放到 [0, 1] 范围内
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
        (255, 127, 0),   # orange
        (106, 180, 193), # blue & green
        (112, 180, 143), # green
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

    plt.xlabel('Model', fontsize=label_fontsize)
    plt.ylabel('Difference Value', fontsize=label_fontsize)
    plt.xticks(x_model_name, model_name, fontsize=16, rotation=45, ha='right')
    plt.yticks([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60], fontsize=tick_fontsize)
    plt.ylim(-0.05, 0.65) # y轴范围
    plt.tight_layout()

    # 保存图片，并设置分辨率为 500 DPI
    plt.savefig("./img/real/SNPC-real_D_cl.pdf", dpi=500)
    plt.savefig("./img/real/SNPC-real_D_cl.png", dpi=500)

    # 显示图形
    plt.show()

    # 输出结果，以表格形式
    print("D_cl in SNPC-real (Mean):")
    print("------------------------------------------------------------------------------")
    print("| Model Name | Value |")
    for i in range(len(model_name)):
        print("| {} | {:.3f} |".format(
            model_name[i],
            mean_D_cl[i] # 保留 3 位小数
        ))
    print("------------------------------------------------------------------------------")

    print("D_cl in SNPC-real (Std):")
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
    1. D_lstd用于衡量单一输入输出语言下的模型输出的平衡性
    2. 不同输入输出语言下的模型输出的差异性：差值矩阵的均值&方差
    3. 热力图 (HeatMap) DeepSeek-V3 模型在不同输入输出语言下的模型输出的差异性
    """
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
