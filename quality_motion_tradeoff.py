import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_csv(
    "results/full_experiment_results.csv"
)


df = df.sort_values("frames")


plt.figure(
    figsize=(7,5),
    dpi=300
)


sizes = df["frames"] * 4


plt.scatter(
    df["mean_motion"],
    df["PSNR"],
    s=sizes,
)



for _, row in df.iterrows():

    plt.annotate(
        f'{int(row["percent"])}%',
        (
            row["mean_motion"],
            row["PSNR"]
        ),
        xytext=(8,8),
        textcoords="offset points",
        fontsize=11
    )


plt.xlabel(
    "Mean Gaussian displacement"
)

plt.ylabel(
    "PSNR (dB)"
)


plt.title(
    "Quality–Motion Trade-off under Temporal Sparsity"
)


plt.grid(True)


os.makedirs(
    "results/final",
    exist_ok=True
)


plt.tight_layout()


plt.savefig(
    "results/final/quality_motion_tradeoff.png",
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    "results/final/quality_motion_tradeoff.pdf",
    bbox_inches="tight"
)


plt.close()


print(
    "saved:"
)
print(
    "results/final/quality_motion_tradeoff.png"
)
print(
    "results/final/quality_motion_tradeoff.pdf"
)
