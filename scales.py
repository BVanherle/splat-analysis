import numpy as np
import torch

from utils import load_models, plot_kde, normalize_array


def plot_scales(model_files: list, model_names: list, normalize_scales=False, reduce="sum"):
    """
    Create a KDE plot to compare the distributions of the scales of the Gaussians of the given models
    The total scale of a Gaussian is the sum of the absolute value of the three scale dimensions
    If normalize_scales is true, the scales of a model are normalized between 0 and 1
    """
    assert reduce in ["sum", "prod"]

    gaussian_models = load_models(model_files)

    # get scales
    scales_data = []

    for model in gaussian_models:
        if reduce == "prod":
            scales = torch.prod(model.get_scaling(), dim=-1).numpy()
        else:
            scales = torch.abs(model.get_scaling()).sum(-1).numpy()

        if normalize_scales:
            scales = normalize_array(scales)

        np.random.shuffle(scales)
        scales_data.append(scales)

    plot_kde(scales_data, model_names, xlabel=f"{reduce} of absolute value of x,y,z scales")


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

    plot_scales(files, names, normalize_scales=True, reduce="sum")