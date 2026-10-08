import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# пути к траекториям
experiments = {
    "beagle_100pct": {
        "frames": 200,
        "percent": 100,
        "PSNR": 33.9677,
        "SSIM": 0.9702466,
        "LPIPS": 0.0547945,
    },
    "beagle_50pct": {
        "frames": 100,
        "percent": 50,
        "PSNR": 33.9266,
        "SSIM": 0.9710606,
        "LPIPS": 0.0518777,
    },
    "beagle_25pct": {
        "frames": 50,
        "percent": 25,
        "PSNR": 32.7479,
        "SSIM": 0.9675056,
        "LPIPS": 0.0558084,
    },
    "beagle_10pct": {
        "frames": 20,
        "percent": 10,
        "PSNR": 28.1411,
        "SSIM": 0.9484873,
        "LPIPS": 0.0743447,
    },
}


results = []


for name, info in experiments.items():

    path = f"motion_results/{name}/trajectory.npy"

    print("loading:", path)

    trajectory = np.load(path)

    # trajectory:
    # time x gaussian x xyz

    num_frames = trajectory.shape[0]
    num_gaussians = trajectory.shape[1]


    # движение между соседними кадрами
    displacement = np.linalg.norm(
        trajectory[1:] - trajectory[:-1],
        axis=2
    )


    # среднее движение одного Gaussian
    mean_motion = displacement.mean()

    # максимальное движение
    max_motion = displacement.max()


    # Gaussian, которые реально двигаются
    gaussian_motion = displacement.max(axis=0)

    moving = gaussian_motion > 1e-4

    moving_count = moving.sum()
    moving_ratio = moving_count / num_gaussians * 100


    row = {
        "experiment": name,
        "frames": info["frames"],
        "percent": info["percent"],
        "gaussians": num_gaussians,
        "PSNR": info["PSNR"],
        "SSIM": info["SSIM"],
        "LPIPS": info["LPIPS"],
        "mean_motion": mean_motion,
        "max_motion": max_motion,
        "moving_gaussians": moving_count,
        "moving_percent": moving_ratio,
    }

    results.append(row)


df = pd.DataFrame(results)

df = df.sort_values(
    by="frames",
    ascending=False
)


os.makedirs(
    "results",
    exist_ok=True
)


df.to_csv(
    "results/full_experiment_results.csv",
    index=False
)


print("\nSaved:")
print("results/full_experiment_results.csv")


# -------------------------
# графики
# -------------------------

os.makedirs(
    "results/figures_motion",
    exist_ok=True
)


plots = [
    ("PSNR", "PSNR"),
    ("mean_motion", "Mean Gaussian Motion"),
    ("max_motion", "Max Gaussian Motion"),
    ("moving_percent", "Moving Gaussian (%)"),
]


for col, ylabel in plots:

    plt.figure(figsize=(6,4))

    plt.plot(
        df["frames"],
        df[col],
        marker="o"
    )

    plt.xlabel("Frames")
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.xticks(
        df["frames"]
    )

    plt.tight_layout()

    plt.savefig(
        f"results/figures_motion/{col}.png",
        dpi=300
    )

    plt.close()


print("Figures saved!")
