import numpy as np

traj=np.load(
    "motion_results/beagle_100pct/trajectory.npy"
)

print(traj.shape)


for t in [0,1,50,100,199]:

    diff=np.linalg.norm(
        traj[t]-traj[0],
        axis=1
    )

    print(
        "time",
        t,
        "mean",
        diff.mean(),
        "max",
        diff.max()
    )
