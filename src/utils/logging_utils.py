import os
import json
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def create_experiment_dir(config_filename=""):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = Path(f"logs/{config_filename}_{timestamp}")
    path.mkdir(parents=True, exist_ok=False)
    return path

def save_results(path, results_dict):
    with open(path / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

def save_distances_csv(path, ecfp_dists, ngf_dists):
    with open(path / "distances.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ecfp_distance", "ngf_distance"])
        writer.writerows(zip(ecfp_dists, ngf_dists))

def save_distance_plot(path, ecfp_dists, ngf_dists, r, title):
    plt.figure(figsize=(5, 5))
    plt.scatter(
        ecfp_dists,
        ngf_dists,
        s=20,
        alpha=0.4,
        edgecolors='black',
        linewidths=0.2,
        facecolor='C0'
    )
    plt.xlabel("Circular fingerprint distances")
    plt.ylabel("Neural fingerprint distances")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.title(f"{title}\n$r={r:.3f}$", fontsize=14)
    plt.tight_layout()
    plt.savefig(path / Path(title+".png"))
    plt.close()
