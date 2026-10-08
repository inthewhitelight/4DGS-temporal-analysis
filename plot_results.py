import csv
import matplotlib.pyplot as plt
import os

os.makedirs("results/figures", exist_ok=True)

frames = []
psnr = []
ssim = []
lpips = []

with open("results_summary.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frames.append(int(row["frames"]))
        psnr.append(float(row["PSNR"]))
        ssim.append(float(row["SSIM"]))
        lpips.append(float(row["LPIPS"]))

# сортировка по количеству кадров
data = sorted(zip(frames, psnr, ssim, lpips), reverse=True)

frames = [x[0] for x in data]
psnr = [x[1] for x in data]
ssim = [x[2] for x in data]
lpips = [x[3] for x in data]


plt.figure(figsize=(6,4))
plt.plot(frames, psnr, marker="o")
plt.xticks(frames)
plt.xlabel("Number of training frames")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs temporal observations")
plt.grid(True)
plt.savefig("results/figures/psnr_vs_frames.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(frames, ssim, marker="o")
plt.xticks(frames)
plt.xlabel("Number of training frames")
plt.ylabel("SSIM")
plt.title("SSIM vs temporal observations")
plt.grid(True)
plt.savefig("results/figures/ssim_vs_frames.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(frames, lpips, marker="o")
plt.xticks(frames)
plt.xlabel("Number of training frames")
plt.ylabel("LPIPS")
plt.title("LPIPS vs temporal observations")
plt.grid(True)
plt.savefig("results/figures/lpips_vs_frames.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done")
