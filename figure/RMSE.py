import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# 基本配置
# =========================================
ZONES = [237, 161, 236, 186, 162, 230, 142, 239, 163]
zone_to_idx = {z: i for i, z in enumerate(ZONES)}
N_ZONES = len(ZONES)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "yellow_tripdata_2025-10.parquet"

BIN = 600
N_BIN = 12
DEGREE = 3


# =========================================
# 计算 avg_counts
# =========================================
def compute_avg_counts(file_path, start_hour, end_hour):
    T_START = start_hour * 3600
    T_END   = end_hour * 3600

    total_counts = np.zeros((N_BIN, N_ZONES, N_ZONES))

    df = pd.read_parquet(file_path)
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"])

    df = df[df["pickup_dt"].dt.weekday < 5]

    sec = (
        df["pickup_dt"].dt.hour * 3600 +
        df["pickup_dt"].dt.minute * 60 +
        df["pickup_dt"].dt.second
    )

    df = df[(sec >= T_START) & (sec < T_END)].copy()

    df = df[
        df["PULocationID"].isin(ZONES) &
        df["DOLocationID"].isin(ZONES)
    ]

    df["bin"] = ((sec.loc[df.index] - T_START) // BIN).astype(int)
    df["date"] = df["pickup_dt"].dt.date

    num_days = df["date"].nunique()

    for _, g in df.groupby("date"):
        for _, r in g.iterrows():
            k = int(r["bin"])
            i = zone_to_idx[r["PULocationID"]]
            j = zone_to_idx[r["DOLocationID"]]
            total_counts[k, i, j] += 1

    avg_counts = total_counts / num_days
    return avg_counts


# =========================================
# MAE
# =========================================
def compute_mae(y, lam):
    return np.mean(np.abs(y - lam))


# =========================================
# Homogeneous MAE
# =========================================
def compute_mae_homogeneous(avg_counts):
    vals = []

    for i in range(N_ZONES):
        for j in range(N_ZONES):
            y = avg_counts[:, i, j]

            if y.sum() < 1e-6:
                continue

            lam = np.full_like(y, y.mean())

            mae = compute_mae(y, lam)
            vals.append(mae)

    return np.array(vals)


# =========================================
# Nonhomogeneous MAE
# =========================================
def compute_mae_nonhomogeneous(avg_counts):
    vals = []
    x = np.arange(N_BIN) + 0.5

    for i in range(N_ZONES):
        for j in range(N_ZONES):
            y = avg_counts[:, i, j]

            if y.sum() < 1e-6:
                continue

            coeff = np.polyfit(x, y, DEGREE)
            lam = np.polyval(coeff, x)

            mae = compute_mae(y, lam)
            vals.append(mae)

    return np.array(vals)


# =========================================
# 画图
# =========================================
def plot_one_period(mae_homo, mae_nonhomo, save_name):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(7,5))

    data = [mae_homo, mae_nonhomo]
    labels = [r'Homogeneous Poisson', r'Nonhomogeneous Poisson']
    colors = ['#0072BD', '#D95319']

    bplot = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color='k', linewidth=1.5),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # scatter
    for i, d in enumerate(data, start=1):
        x = np.random.normal(i, 0.06, size=len(d))
        ax.scatter(x, d, color=colors[i-1], edgecolors='k', s=18, alpha=0.7)

    ax.set_ylabel(r'MAE')
    ax.set_xticklabels(labels, fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color('#E6E6E6')

    ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


# =========================================
# 主流程
# =========================================
print("Processing 6–8...")
avg_6_8 = compute_avg_counts(DATA_FILE, 6, 8)
mae_homo_6_8 = compute_mae_homogeneous(avg_6_8)
mae_nonhomo_6_8 = compute_mae_nonhomogeneous(avg_6_8)

print("Processing 17–19...")
avg_17_19 = compute_avg_counts(DATA_FILE, 17, 19)
mae_homo_17_19 = compute_mae_homogeneous(avg_17_19)
mae_nonhomo_17_19 = compute_mae_nonhomogeneous(avg_17_19)

# =========================================
# 画图
# =========================================
plot_one_period(
    mae_homo_6_8, mae_nonhomo_6_8,
    save_name="MAE_6_8.png"
)

plot_one_period(
    mae_homo_17_19, mae_nonhomo_17_19,
    save_name="MAE_17_19.png"
)