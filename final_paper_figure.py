import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================
# Load results
# =========================

df = pd.read_csv(
    "results/full_experiment_results.csv"
)

df = df.sort_values(
    "frames"
)


frames = df["frames"]


# =========================
# Paper style
# =========================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


fig, axes = plt.subplots(
    2,
    2,
    figsize=(10,7),
    dpi=300
)


# =========================
# (a) Reconstruction quality
# =========================

ax = axes[0,0]

ax.plot(
    frames,
    df["PSNR"],
    marker="o",
    linewidth=2,
    label="PSNR"
)

ax.set_xlabel("Frames")
ax.set_ylabel("PSNR (dB)")

ax.grid(True)

ax.set_title("(a) Rendering quality")



# =========================
# (b) Perceptual metrics
# =========================

ax = axes[0,1]

ax.plot(
    frames,
    df["SSIM"],
    marker="o",
    linewidth=2,
    label="SSIM"
)


ax.plot(
    frames,
    df["LPIPS"],
    marker="s",
    linewidth=2,
    label="LPIPS"
)


ax.set_xlabel("Frames")
ax.set_ylabel("Metric")

ax.grid(True)

ax.legend()

ax.set_title("(b) Image similarity")



# =========================
# (c) Motion magnitude
# =========================

ax = axes[1,0]


ax.plot(
    frames,
    df["mean_motion"],
    marker="o",
    linewidth=2,
    label="Mean displacement"
)


ax.plot(
    frames,
    df["max_motion"],
    marker="s",
    linewidth=2,
    label="Maximum displacement"
)


ax.set_xlabel("Frames")
ax.set_ylabel("Displacement")


ax.grid(True)

ax.legend()

ax.set_title("(c) Gaussian motion")



# =========================
# (d) Model complexity
# =========================

ax = axes[1,1]


ax.plot(
    frames,
    df["gaussians"],
    marker="o",
    linewidth=2
)


ax.set_xlabel("Frames")

ax.set_ylabel(
    "Gaussian points"
)


ax.grid(True)

ax.set_title(
    "(d) Representation size"
)



# =========================
# Save
# =========================

os.makedirs(
    "results/final",
    exist_ok=True
)


plt.tight_layout()


plt.savefig(
    "results/final/experiment_paper.png",
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    "results/final/experiment_paper.pdf",
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:"
)
print(
    "results/final/experiment_paper.png"
)
print(
    "results/final/experiment_paper.pdf"
)
