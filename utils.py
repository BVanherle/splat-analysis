import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gaussian_model import GaussianModel


def load_models(model_files):
    gaussian_models = []

    for model_file in model_files:
        gaussian_model = GaussianModel(sh_degree=3)
        gaussian_model.load_ply(model_file)
        gaussian_model.requires_grad_(False)
        gaussian_models.append(gaussian_model)

    return gaussian_models


def normalize_array(array: np.array):
    min_val = array.min()
    max_val = array.max()
    normalized_arr = (array - min_val) / (max_val - min_val)
    return normalized_arr


def plot_kde(datasets, labels, xlabel):
    length = min([len(w) for w in datasets])

    # collect in dataframe
    data_dict = {}
    for name, data in zip(labels, datasets):
        data_dict[name] = data[:length]
    df = pd.DataFrame(data_dict)

    sns.violinplot(df)
    plt.xlabel(xlabel)
    plt.show()
