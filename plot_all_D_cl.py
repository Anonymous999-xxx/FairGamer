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
    # List of image names
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

    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    # Set overall layout parameters
    plt.subplots_adjust(wspace=0.1, hspace=-0.6)  # Adjust subplot spacing

    # Iterate through all subplots to add images and titles
    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):  # Ensure we don't exceed the number of images
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            # ax.set_title(titles[i], fontsize=16, y=-0.15, pad=10)  # title under the subplot
            ax.set_title(titles[i], fontsize=14)
            ax.axis('off')  # Turn off axes
        else:
            ax.axis('off')  # Hide extra subplots if there are more than images
    
    plt.tight_layout()

    # Save the image
    plt.savefig('./img/All_D_cl.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to prevent memory leaks

    print("image saved in './img/All_D_cl.pdf', DPI is 500.")