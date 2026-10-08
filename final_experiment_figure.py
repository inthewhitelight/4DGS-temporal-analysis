import pandas as pd
import matplotlib.pyplot as plt
import os


# =====================
# load data
# =====================

df = pd.read_csv(
    "results/full_experiment_results.csv"
)

# sort by frames
df = df.sort_values("frames")


frames = df["frames"]


# =====================
# create figure
# =====================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,9),
    dpi=300
)


# =====================
# (a) Quality
# =====================

ax = axes[0,0]

ax.plot(
    frames,
    df["PSNR"],
    marker="o",
    label="PSNR (dB)"
)

ax2 = ax.twinx()

ax2.plot(
    frames,
    df["SSIM"],
    marker="s",
    label="SSIM"
)

ax2.plot(
    frames,
    df["LPIPS"],
    marker="^",
    label="LPIPS"
)


ax.set_xlabel("Number of frames")
ax.set_ylabel("PSNR")
ax2.set_ylabel("SSIM / LPIPS")

ax.set_title("(a) Reconstruction quality")

ax.grid(True)


# объединяем легенды

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    lines+lines2,
    labels+labels2,
    fontsize=8
)



# =====================
# (b) Motion
# =====================

ax = axes[0,1]


ax.plot(
    frames,
    df["mean_motion"],
    marker="o",
    label="Mean motion"
)


ax.plot(
    frames,
    df["max_motion"],
    marker="s",
    label="Max motion"
)


ax.set_xlabel("Number of frames")
ax.set_ylabel("Motion magnitude")

ax.set_title("(b) Gaussian motion")

ax.grid(True)
ax.legend()



# =====================
# (c) Gaussian count
# =====================

ax = axes[1,0]


ax.plot(
    frames,
    df["gaussians"],
    marker="o"
)


ax.set_xlabel("Number of frames")
ax.set_ylabel("Gaussian points")

ax.set_title("(c) Gaussian representation size")

ax.grid(True)



# =====================
# (d) Moving ratio
# =====================

ax = axes[1,1]


ax.plot(
    frames,
    df["moving_percent"],
    marker="o"
)


ax.set_xlabel("Number of frames")
ax.set_ylabel("Moving Gaussians (%)")


ax.set_ylim(
    95,
    100
)

ax.set_title("(d) Active deformation ratio")

ax.grid(True)



# =====================
# save
# =====================

plt.tight_layout()


os.makedirs(
    "results/final",
    exist_ok=True
)


plt.savefig(
    "results/final/experiment_summary.png",
    bbox_inches="tight"
)


plt.close()


print(
    "saved:",
    "results/final/experiment_summary.png"
)
