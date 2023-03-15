import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set default font for graphs
font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 15}


def make_folder(name):
    """
    Makes a folder with given name if it does not already exist

    Input: name of the folder to be created
    Output: None
    """
    if not os.path.exists(name):
        os.makedirs(name)


def plot_classification(config, auroc, aupr):
    """
    Creates boxplots of the AUROC and AUPR scores and saves them in the "figures" folder under the experiment's name.

    Input:
        config: dictionary config file
        auroc: dictionary of auroc scores
        aupr: dictionary of aupr scores
    Output: None
    """
    folder = "figures/" + config["experiment_name"]
    title = config["experiment_title"]
    make_folder(folder)
    file = folder + "/" + config["experiment_name"]
    plot_boxplot(title + " AUROC", file + "_AUROC.png", auroc)
    plot_boxplot(title + " AUPR", file + "_AUPR.png", aupr)


def plot_regression(config, mses):
    """
    Creates boxplots of the MSE scores and saves them in the "figures" folder under the experiment's name.

    Input:
        config: dictionary config file
        mses: dictionary of mean squared error scores
    Output: None
    """
    folder = "figures/" + config["experiment_name"]
    title = config["experiment_title"]
    make_folder(folder)
    file = folder + "/" + config["experiment_name"]
    plot_boxplot(title + " MSE", file + "_MSE.png", mses)


def plot_boxplot(title, file_name, data_dict):
    """
    Creates a boxplot with a title and axes.

    Inputs:
        title: string of the title of the plot
        file_name: name of the file the plot will be saved in
        data_dict: dictionary of data to be plotted
    Output: matlibplot figure object of the boxplot
    """
    data = list(data_dict.values())
    ticks = list(data_dict.keys())

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    ax.boxplot(data)
    ax.set_xticklabels(ticks)
    plt.title(title)
    plt.savefig(file_name)
    plt.close(fig)
    return plt
