import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


experiments = [
    "beagle_100pct",
    "beagle_50pct",
    "beagle_25pct",
    "beagle_10pct"
]


max_points = 5000


output_dir = "results/figures/motion_maps"

os.makedirs(
    output_dir,
    exist_ok=True
)


for exp in experiments:

    print("\nProcessing:", exp)

    trajectory_path = (
        f"motion_results/{exp}/trajectory.npy"
    )


    trajectory = np.load(
        trajectory_path
    )


    print(
        "trajectory:",
        trajectory.shape
    )


    start = trajectory[0]


    motion = np.linalg.norm(
        trajectory - start[None,:,:],
        axis=-1
    )


    motion_mag = motion.max(axis=0)


    xyz = trajectory[0]


    print(
        "mean motion:",
        motion_mag.mean()
    )

    print(
        "max motion:",
        motion_mag.max()
    )


    if len(xyz) > max_points:

        idx = np.random.choice(
            len(xyz),
            max_points,
            replace=False
        )

        xyz = xyz[idx]
        motion_mag = motion_mag[idx]


    fig = plt.figure(
        figsize=(8,7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    scatter = ax.scatter(
        xyz[:,0],
        xyz[:,1],
        xyz[:,2],
        c=motion_mag,
        s=3,
        cmap="turbo"
    )


    ax.set_title(
        exp + "\nGaussian motion magnitude"
    )


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.6,
        label="Motion"
    )


    save_file = (
        f"{output_dir}/{exp}.png"
    )


    plt.tight_layout()

    plt.savefig(
        save_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "saved:",
        save_file
    )


print("\nDone!")
