import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


csv_path = "results/full_experiment_results.csv"

save_path = "results/final_experiment_summary.png"


# ==========================
# Load data
# ==========================

df = pd.read_csv(csv_path)


order = [
    "beagle_100pct",
    "beagle_50pct",
    "beagle_25pct",
    "beagle_10pct"
]


df["experiment"] = pd.Categorical(
    df["experiment"],
    categories=order,
    ordered=True
)

df = df.sort_values("experiment")


labels = [
    "100%",
    "50%",
    "25%",
    "10%"
]


x = np.arange(len(df))


# ==========================
# Figure
# ==========================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)


# ---- PSNR ----

axes[0,0].plot(
    x,
    df["PSNR"],
    marker="o",
    linewidth=2
)

axes[0,0].set_title(
    "Rendering quality (PSNR)"
)

axes[0,0].set_ylabel(
    "dB"
)

axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(labels)

axes[0,0].grid(True)


# ---- Gaussian number ----

axes[0,1].plot(
    x,
    df["gaussians"],
    marker="o",
    linewidth=2
)

axes[0,1].set_title(
    "Number of Gaussian primitives"
)

axes[0,1].set_ylabel(
    "Count"
)

axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(labels)

axes[0,1].grid(True)


# ---- Motion ----

axes[1,0].plot(
    x,
    df["mean_motion"],
    marker="o",
    linewidth=2
)

axes[1,0].set_title(
    "Mean temporal deformation"
)

axes[1,0].set_ylabel(
    "Displacement"
)

axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(labels)

axes[1,0].grid(True)


# ---- Moving Gaussian percentage ----

axes[1,1].plot(
    x,
    df["moving_percent"],
    marker="o",
    linewidth=2
)

axes[1,1].set_title(
    "Active Gaussian ratio"
)

axes[1,1].set_ylabel(
    "%"
)

axes[1,1].set_ylim(
    95,
    100
)

axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(labels)

axes[1,1].grid(True)


# ==========================
# Layout
# ==========================

plt.suptitle(
    "4D Gaussian Splatting temporal sampling analysis",
    fontsize=16
)


plt.tight_layout()


os.makedirs(
    os.path.dirname(save_path),
    exist_ok=True
)


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


print(
    "Saved:",
    save_path
)

