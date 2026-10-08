import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os


experiments = [
    "beagle_100pct",
    "beagle_50pct",
    "beagle_25pct",
    "beagle_10pct"
]


titles = [
    "100% frames",
    "50% frames",
    "25% frames",
    "10% frames"
]


trajectories = {}

for exp in experiments:
    path = f"motion_results/{exp}/trajectory.npy"
    print("loading", path)

    trajectories[exp] = np.load(path)


# одинаковый масштаб для всех
all_points = np.concatenate(
    [
        t.reshape(-1,3)
        for t in trajectories.values()
    ],
    axis=0
)


xmin, xmax = all_points[:,0].min(), all_points[:,0].max()
ymin, ymax = all_points[:,1].min(), all_points[:,1].max()
zmin, zmax = all_points[:,2].min(), all_points[:,2].max()


# количество кадров
max_frames = max(
    t.shape[0]
    for t in trajectories.values()
)


os.makedirs(
    "results/comparison_frames",
    exist_ok=True
)


frames = []


for frame in range(max_frames):

    fig = plt.figure(
        figsize=(12,10)
    )


    for idx, exp in enumerate(experiments):

        ax = fig.add_subplot(
            2,
            2,
            idx+1,
            projection="3d"
        )


        traj = trajectories[exp]


        # если у эксперимента меньше кадров
        if frame >= len(traj):
            current = traj[-1]
        else:
            current = traj[frame]


        ax.scatter(
            current[:,0],
            current[:,1],
            current[:,2],
            s=0.3
        )


        ax.set_title(
            titles[idx],
            fontsize=14
        )


        ax.set_xlim(
            xmin,
            xmax
        )

        ax.set_ylim(
            ymin,
            ymax
        )

        ax.set_zlim(
            zmin,
            zmax
        )


        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


    plt.suptitle(
        "4D Gaussian Splatting temporal motion comparison",
        fontsize=18
    )


    plt.tight_layout()


    filename = (
        f"results/comparison_frames/{frame:04d}.png"
    )


    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()


    frames.append(
        imageio.imread(filename)
    )


imageio.mimsave(
    "results/beagle_motion_comparison.mp4",
    frames,
    fps=20
)


print(
    "saved:",
    "results/beagle_motion_comparison.mp4"
)
