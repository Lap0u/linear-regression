import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import pandas as pd


def count(array):
    return len(array)


def load_data(filename: str, target_col_name: str, t: callable):
    df = (
        pd.read_csv(filename, sep=",", converters={target_col_name: t})
        .drop(["Index"], axis=1)
        .select_dtypes(include=["float64", "int64"])
    )
    y = df[target_col_name]
    x = df.drop([target_col_name], axis=1)
    return x, y


def mean(array):
    return sum(array) / len(array)


def min_(array):
    min_ = array[0]
    for i in range(len(array)):
        if array[i] < min_:
            min_ = array[i]
    return min_


def quantile(array, q):
    return array[int(len(array) * q)]


def remove_empty_fields(array):
    x = [float(i) for i in array if not math.isnan(i)]
    return pd.DataFrame(x)


def max_(array):
    max_ = array[0]
    for i in range(len(array)):
        if array[i] > max_:
            max_ = array[i]
    return max_


def std(array, mean):
    sum_ = 0
    for i in array:
        sum_ += (i - mean) ** 2
    return (sum_ / len(array)) ** 0.5


def is_valid_path(file_path):
    if path.isfile(file_path) == False:
        raise Exception("File does not exist")
    if (os.access(file_path, os.R_OK)) == False:
        raise Exception("File is not readable")
    if Path(file_path).suffix != ".csv":
        raise Exception("File is not a csv file")


def plot_data(x, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(x[positive, 0], x[positive, 1], "k+", label=pos_label)
    plt.plot(x[negative, 0], x[negative, 1], "yo", label=neg_label)


def map_feature(x1, x2):
    """
    Feature mapping function to polynomial features
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1 ** (i - j) * (x2**j)))
    return np.stack(out, axis=1)


def plot_decision_boundary(w, b, x, y):
    # Credit to dibgerge on Github for this plotting code

    plot_data(x[:, 0:2], y)

    if x.shape[1] <= 2:
        plot_x = np.array([min(x[:, 0]), max(x[:, 0])])
        plot_y = (-1.0 / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid_(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0.5
        plt.contour(u, v, z, levels=[0.5], colors="g")


def heat_map(df):
    sns.heatmap(df, annot=True)
    plt.show()


def normalize_array(x):
    return (x - x.min()) / (x.max() - x.min())


def denormalize_array(list, elem):
    return (elem * (max(list) - min(list))) + min(list)


def sigmoid_(z):
    return 1 / (1 + np.exp(-z))
