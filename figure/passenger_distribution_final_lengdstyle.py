import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
})

# ============================
# 文件路径
# ============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_10 = BASE_DIR / "data" / "yellow_tripdata_2025-10.parquet"
DATA_PATH_11 = BASE_DIR / "data" / "yellow_tripdata_2025-11.parquet"

zones = [237, 161, 236, 186, 162, 230, 142, 239, 163]

# ============================
# 数据处理
# ============================
def process_file(path):
    df = pd.read_parquet(path)

    df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], unit='ms')

    df = df[df["pickup_datetime"].dt.weekday < 5]

    df = df[
        df["PULocationID"].isin(zones) &
        df["DOLocationID"].isin(zones)
    ]

    df["hour"] = df["pickup_datetime"].dt.hour
    df["date"] = df["pickup_datetime"].dt.date

    hourly_avg = (
        df.groupby(["date", "hour"])
        .size()
        .groupby("hour")
        .mean()
    )

    return hourly_avg.reindex(range(24), fill_value=0)

h_10 = process_file(DATA_PATH_10)
h_11 = process_file(DATA_PATH_11)

# ============================
# 绘图
# ============================
hours = np.arange(24)

fig, ax = plt.subplots(figsize=(16, 5.6), dpi=150)

line_oct, = ax.plot(hours, h_10, color='#4575B4', linewidth=2.5, label=r"October")
line_nov, = ax.plot(hours, h_11, color='#D73027', linewidth=2.5, linestyle='--', label=r"November")

# ============================
# 研究区间（无填充细框）
# ============================
y_min, y_max = ax.get_ylim()

rect1 = Rectangle((6, y_min), 2, y_max - y_min,
                  linewidth=1.5, edgecolor='#4575B4',
                  facecolor='none', linestyle=':')

rect2 = Rectangle((17, y_min), 2, y_max - y_min,
                  linewidth=1.5, edgecolor='#D73027',
                  facecolor='none', linestyle=':')

ax.add_patch(rect1)
ax.add_patch(rect2)

# ============================
# 坐标轴
# ============================
ax.set_xlim(0, 23)
ax.set_xticks(np.arange(0, 24, 1))
ax.tick_params(axis='x', labelsize=9)

ax.set_xlabel(r"Hour of Day")
ax.set_ylabel(r"Number of Passengers")

# 风格
ax.grid(True, axis='y', linestyle='--', alpha=0.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================
# 图例（关键改动🔥）
# ============================
legend_elements = [
    line_oct,
    line_nov,
    Line2D([0], [0], color='#4575B4', linestyle=':', linewidth=1.5, label='Morning Peak Period (6-8 AM)'),
    Line2D([0], [0], color='#D73027', linestyle=':', linewidth=1.5, label='Evening Peak Period (5-7 PM)'),
]

ax.legend(handles=legend_elements, frameon=True)

plt.tight_layout()

plt.show()