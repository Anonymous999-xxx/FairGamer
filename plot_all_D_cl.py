import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

# Setting colored output:
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x

if __name__ == '__main__':
    # 图片名称列表
    task_names = [
        'SNPC-real', 
        'SNPC-virtual', 
        'ICO-real', 
        'ICO-virtual', 
        'GGS-real', 
        'GGS-virtual'
    ]
    image_paths = [f'./img/{name}_D_cl.png' for name in task_names]
    titles = ['(a) SNPC-Real', '(b) SNPC-Virtual', '(c) ICO-Real', '(d) ICO-Virtual', '(e) GGS-Real', '(f) GGS-Virtual']

    # 创建3行2列的子图
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    # 设置整体布局参数
    plt.subplots_adjust(wspace=0.1, hspace=-0.6)  # 调整子图间距

    # 遍历所有子图并添加图片和标题
    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):  # 确保不超出图片数量范围
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            # ax.set_title(titles[i], fontsize=16, y=-0.15, pad=10)  # 将标题放在子图下方
            ax.set_title(titles[i], fontsize=14)
            ax.axis('off')  # 关闭坐标轴
        else:
            ax.axis('off')  # 如果子图多于图片数量，隐藏多余子图
    
    plt.tight_layout()

    # 保存图片
    plt.savefig('./img/All_D_cl.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()  # 关闭图形，避免内存泄漏

    print("image saved in './img/All_D_cl.pdf', DPI is 500.")