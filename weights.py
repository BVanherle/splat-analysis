import numpy as np
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gaussian_model import GaussianModel
from utils import load_models, plot_kde


def plot_weights(model_files: list, model_names: list):
    gaussian_models = load_models(model_files)

    # get weights
    weight_data = []

    for model in gaussian_models:
        weights = torch.abs(model._features_rest).sum(-1).sum(-1).numpy()
        np.random.shuffle(weights)
        weight_data.append(weights)

    plot_kde(weight_data, model_names, xlabel="Sum of absolute values of non DC weights.")


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

plot_weights(files, names)