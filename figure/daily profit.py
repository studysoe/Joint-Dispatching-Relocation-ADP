import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================================
# 全局风格
# =========================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.5,
})

# =========================================
# 数据
# =========================================
fluid_approx_old = np.array([
    51020.77,48173.44,54491.61,57891.07,49724.47,50880.80,
    51093.61,52732.67,58895.52,53654.21,46991.92,58357.52,
    55014.37,60791.11,53653.54,47049.35,58596.74
])

unhomogeneous_rl_old = np.array([
    50655.10,49242.56,56009.11,58423.35,52182.31,49979.90,
    53488.19,52754.86,59306.72,55402.91,48749.34,57972.83,
    55440.75,61253.95,55522.95,49002.36,59086.78
])

fluid_approx_new = np.array([
    7259.60,6112.20,6969.16,8942.08,7335.14,6529.37,
    7294.79,7987.32,7352.46,9285.88,8111.38,7317.30,
    10834.77,7405.92,7197.45,5359.93,5599.17
])

unhomogeneous_rl_new = np.array([
    8025.11,6364.08,7622.13,8647.19,7805.70,6864.42,
    7847.01,8662.57,7646.09,10388.68,8743.76,7660.89,
    11742.73,8577.06,7952.79,5851.91,6795.16
])

# =========================================
# 子图函数
# =========================================
def draw_box(ax, fluid, rl):

    color = '#4575B4'
    data_list = [fluid, rl]
    labels = [r'FA-HPP', r'ADP-NHPP']
    positions = [1, 2]

    stats = []
    for data in data_list:
        stats.append({
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'median': np.percentile(data, 50),
            'mean': np.mean(data),
            'p10': np.percentile(data, 10),
            'p90': np.percentile(data, 90),
            'p5': np.percentile(data, 5),
            'p95': np.percentile(data, 95),
        })

    for i, s in enumerate(stats):
        x = positions[i]

        ax.add_patch(plt.Rectangle(
            (x-0.2, s['q1']),
            0.4,
            s['q3'] - s['q1'],
            facecolor=color,
            alpha=0.5,
            edgecolor='black',
            linewidth=1.5
        ))

        ax.plot([x-0.2, x+0.2], [s['median'], s['median']],
                color='black', linewidth=1.5)

        ax.plot([x, x], [s['p10'], s['p90']], color='black', linewidth=1.5)
        ax.plot([x-0.1, x+0.1], [s['p10'], s['p10']], color='black', linewidth=1.5)
        ax.plot([x-0.1, x+0.1], [s['p90'], s['p90']], color='black', linewidth=1.5)

        ax.scatter(x, s['p5'], marker='^', color='black', zorder=3)
        ax.scatter(x, s['p95'], marker='^', color='black', zorder=3)

    means = [s['mean'] for s in stats]
    ax.plot(positions, means, linestyle='--', color='black', marker='o')

    for i, data in enumerate(data_list):
        x = np.random.normal(positions[i], 0.04, size=len(data))
        ax.scatter(x, data, color='gray', alpha=0.3, s=10)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color('#E6E6E6')

    ax.tick_params(axis='both', length=0)

# =========================================
# 主图
# =========================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# ✅ 修正：小数值是 morning
draw_box(axes[0], fluid_approx_new, unhomogeneous_rl_new)
draw_box(axes[1], fluid_approx_old, unhomogeneous_rl_old)

axes[0].set_ylabel(r'Profit')

# =========================================
# 子图标签（放下面）
# =========================================
axes[0].text(0.5, -0.1, r'(a) Morning peak period',
             transform=axes[0].transAxes,
             ha='center', va='top')

axes[1].text(0.5, -0.1, r'(b) Evening peak period',
             transform=axes[1].transAxes,
             ha='center', va='top')

# =========================================
# 共享图例
# =========================================
legend_elements = [
    Patch(facecolor='#4575B4', alpha=0.5, edgecolor='black', label='25\\%–75\\%'),
    Line2D([0], [0], color='black', lw=1.5, label='10\\%–90\\%'),
    Line2D([0], [0], color='black', lw=1.5, label='Median'),
    Line2D([0], [0], marker='o', color='black', linestyle='--', label='Mean'),
    Line2D([0], [0], marker='^', color='black', linestyle='None', label='5\\%, 95\\%'),
]

fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.08),  # 调整y坐标控制高度
           ncol=5,
           frameon=True)

plt.subplots_adjust(bottom=0.28, wspace=0.3)

plt.show()