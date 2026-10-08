import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


trajectory_path = (
    "motion_results/beagle_100pct/trajectory.npy"
)

out_dir = (
    "results/motion_color_frames"
)

video_path = (
    "results/beagle_motion_color.mp4"
)


os.makedirs(out_dir, exist_ok=True)


trajectory = np.load(trajectory_path)

T, N, _ = trajectory.shape

print(trajectory.shape)


# выбираем точки
max_points = 8000

np.random.seed(0)

idx = np.random.choice(
    N,
    min(N,max_points),
    replace=False
)

trajectory = trajectory[:,idx]


# движение каждой точки
motion = np.linalg.norm(
    trajectory - trajectory[0][None,:,:],
    axis=-1
)


# максимальное движение
motion_max = motion.max(axis=0)


xyz_all = trajectory.reshape(-1,3)

mins = xyz_all.min(axis=0)
maxs = xyz_all.max(axis=0)


frames=[]


for t in range(T):

    print(t)

    xyz = trajectory[t]


    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    sc=ax.scatter(
        xyz[:,0],
        xyz[:,1],
        xyz[:,2],
        c=motion_max,
        cmap="coolwarm",
        s=2
    )


    ax.set_xlim(mins[0],maxs[0])
    ax.set_ylim(mins[1],maxs[1])
    ax.set_zlim(mins[2],maxs[2])


    ax.axis("off")

    ax.set_title(
        f"Gaussian deformation frame {t}"
    )


    plt.colorbar(
        sc,
        shrink=0.5,
        label="Displacement"
    )


    file=f"{out_dir}/frame_{t:04d}.png"


    plt.savefig(
        file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    frames.append(
        imageio.imread(file)
    )


imageio.mimsave(
    video_path,
    frames,
    fps=20
)


print("saved", video_path)
