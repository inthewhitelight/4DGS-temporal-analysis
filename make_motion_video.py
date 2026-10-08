import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


exp = "beagle_10pct"

traj = np.load(
    f"motion_results/{exp}/trajectory.npy"
)

print(traj.shape)


out_dir = "results/video_frames"
os.makedirs(
    out_dir,
    exist_ok=True
)


frames = []


# фиксируем масштаб камеры
xyz0 = traj[0]

xmin, xmax = xyz0[:,0].min(), xyz0[:,0].max()
ymin, ymax = xyz0[:,1].min(), xyz0[:,1].max()
zmin, zmax = xyz0[:,2].min(), xyz0[:,2].max()


for i in range(len(traj)):

    xyz = traj[i]


    fig = plt.figure(
        figsize=(6,6)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(
        xyz[:,0],
        xyz[:,1],
        xyz[:,2],
        s=0.5
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


    ax.set_title(
        f"{exp} frame {i}"
    )


    path = f"{out_dir}/{i:04d}.png"

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    frames.append(
        imageio.imread(path)
    )


imageio.mimsave(
    f"results/{exp}_motion.mp4",
    frames,
    fps=20
)


print("saved video")
