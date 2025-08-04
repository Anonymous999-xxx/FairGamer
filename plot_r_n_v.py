import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data: (x, y) pairs for eight models across three datasets
models = [
    "GPT-4o", "Grok-3", "Grok-3-Mini", "DeepSeek-V3",
    "Qwen-2.5-72B", "Qwen-2.5-7B", "LLaMA-3.1-70B", "LLaMA-3.1-8B"
]

group1 = np.array([
    [0.167, 0.131],
    [0.240, 0.073],
    [0.293, 0.178],
    [0.237, 0.195],
    [0.060, 0.018],
    [0.148, 0.082],
    [0.178, 0.108],
    [0.118, 0.108],
])

group2 = np.array([
    [0.093, 0.092],
    [0.171, 0.529],
    [0.150, 0.209],
    [0.171, 0.293],
    [0.101, 0.119],
    [0.202, 0.166],
    [0.204, 0.281],
    [0.218, 0.211],
])

group3 = np.array([
    [0.572, 0.370],
    [0.797, 0.666],
    [0.753, 0.813],
    [0.440, 0.155],
    [0.708, 0.610],
    [0.517, 0.340],
    [0.633, 0.357],
    [0.489, 0.211],
])

# Configurations
groups = [group1, group2, group3]
colors = ['tab:red', 'tab:green', 'tab:blue']         # dataset colors
markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'v']     # unique marker per model
labels_datasets = ['SNPC', 'ICO', 'GGS']
marker_size = 150                                      # marker size
line_width = 3                                         # thicker regression lines

# Plot
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Scatter points: loop over models to keep marker consistent across datasets
for idx_model, marker in enumerate(markers):
    for color, g in zip(colors, groups):
        x, y = g[idx_model]
        ax.scatter(x, y,
                    color=color,
                    marker=marker,
                    s=marker_size,
                    edgecolors='k',
                    linewidths=1.0,
                    alpha=0.85)

# Linear fits (one per dataset)
for g, color, label in zip(groups, colors, labels_datasets):
    x_vals, y_vals = g[:, 0], g[:, 1]
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_fit,
            slope * x_fit + intercept,
            color=color,
            linestyle='--',
            linewidth=line_width,
            label=label)

# Aesthetics
ax.set_xlabel('Real', fontsize=30)
ax.set_ylabel('Virtual', fontsize=30)
# Removed title as requested
ax.grid(True, linestyle=':')
ax.tick_params(axis='both', which='major', labelsize=16)  # tick fontsize

# Legends
# --- Tasks legend (datasets) ---
legend_tasks = ax.legend(title='Tasks', fontsize=24, title_fontsize=24,
                         loc='upper left', bbox_to_anchor=(1.02, 0.25))
ax.add_artist(legend_tasks)

# --- Model legend (markers) ---
model_handles = [Line2D([], [], marker=markers[i], linestyle='None', color='k',
                        markeredgecolor='k', markersize=np.sqrt(marker_size), label=models[i])
                 for i in range(len(models))]
legend_models = ax.legend(handles=model_handles, title='Models', fontsize=24, title_fontsize=24,
                          loc='upper left', bbox_to_anchor=(1.02, 1))

# Layout & save
plt.tight_layout()
plt.savefig('plot.pdf', dpi=500)
plt.show()
