import numpy as np
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gaussian_model import GaussianModel


def plot_weights(model_files: list, model_names: list):
    gaussian_models = []

    # load models
    for model_file in model_files:
        gaussian_model = GaussianModel(sh_degree=3)
        gaussian_model.load_ply(model_file)
        gaussian_model.requires_grad_(False)
        gaussian_models.append(gaussian_model)

    # get weights
    weight_data = []

    for model in gaussian_models:
        weights = torch.abs(model._features_rest).sum(-1).sum(-1).numpy()
        np.random.shuffle(weights)
        weight_data.append(weights)

    # find number of splats for smallest model
    length = min([len(w) for w in weight_data])

    # collect in dataframe
    data_dict = {}
    for name, weights in zip(model_names, weight_data):
        data_dict[name] = weights[:length]
    df = pd.DataFrame(data_dict)

    sns.kdeplot(df)
    plt.xlabel("Sum of absolute values of non DC weights.")
    plt.show()


if __name__ == "__main__":
    files = [
        "data/2eca5280-3/point_cloud/iteration_30000/point_cloud.ply",
        "data/2eb21784-a/point_cloud/iteration_30000/point_cloud.ply"
    ]

    names = ["plant", "vase"]

    plot_weights(files, names)