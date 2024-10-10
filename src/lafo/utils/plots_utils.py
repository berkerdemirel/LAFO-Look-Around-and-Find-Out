import matplotlib.pyplot as plt
import numpy as np


def plot_helper(score_in: np.ndarray, scores_out_test: np.ndarray, in_dataset: str, ood_dataset_name: str):
    plt.hist(score_in, bins=70, alpha=0.5, label="in", density=True)
    plt.hist(scores_out_test, bins=70, alpha=0.5, label="out", density=True)
    plt.legend(loc="upper right", fontsize=9)
    if ood_dataset_name == "dtd":
        ood_dataset_name = "Texture"
    print(ood_dataset_name)
    # plt.xlabel('LAFO(x)', fontsize=16)
    plt.xlabel("LAFO(x)", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(f"{in_dataset} vs {ood_dataset_name}", fontsize=16)
    plt.savefig(f"./{in_dataset}_{ood_dataset_name}_hist_LAFO.png", dpi=600)
    plt.close()
