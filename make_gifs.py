import imageio.v2 as imageio
import os


videos = [
    "results/beagle_100pct_motion.mp4",
    "results/beagle_50pct_motion.mp4",
    "results/beagle_25pct_motion.mp4",
    "results/beagle_10pct_motion.mp4",
    "results/beagle_motion_comparison.mp4",
]


for video in videos:

    reader = imageio.get_reader(video)

    name = os.path.splitext(video)[0]
    output = name + ".gif"

    frames = []

    for i, frame in enumerate(reader):

        # берём каждый второй кадр для уменьшения размера
        if i % 2 == 0:
            frames.append(frame)


    imageio.mimsave(
        output,
        frames,
        fps=10
    )

    print("saved:", output)
