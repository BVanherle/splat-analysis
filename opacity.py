import numpy as np
import torch

from utils import load_models, plot_kde


def plot_opacity(model_files: list, model_names: list):
    gaussian_models = load_models(model_files)

    # get opacity values
    weight_data = []

    for model in gaussian_models:
        weights = torch.squeeze(model.get_opacity()).numpy()
        np.random.shuffle(weights)
        weight_data.append(weights)

    plot_kde(weight_data, model_names, xlabel="Splat opacity values")


if __name__ == "__main__":
    files = [
        "data/scenes/bicycle.ply",
        "data/scenes/counter.ply",
        "data/scenes/drjohnson.ply",
        "data/scenes/kitchen.ply",
        "data/scenes/train.ply",
        "data/scenes/truck.ply",
    ]

    names = ["bicycle", "counter", "drjohnson", "kitchen", "train", "truck"]

    plot_opacity(files, names)