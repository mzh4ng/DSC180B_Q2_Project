import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


def create_pca(data, targets, target_column):
    """
    """
    # idx = np.random.permutation(data.index)
    # data = data.reindex(idx)[:500]
    # targets = targets.reindex(idx)[:500]
    # # Run MDS
    # ## 
    # from sklearn.metrics import pairwise_distances
    # from sklearn.manifold import MDS

    # import warnings
    # from sklearn.exceptions import DataConversionWarning
    # warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # distance_metric = 'jaccard'

    # distances = pairwise_distances(data.to_numpy(), metric=distance_metric)
    # pd.DataFrame(distances).to_csv('distances.csv')
    # mds = MDS(dissimilarity='precomputed', random_state=0, normalized_stress=False)
    # pca_data = mds.fit_transform(distances)

    # Run PCA
    ## Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    ## Run sklearn PCA on data
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    # Convert ndarray to DataFrame, Add targets back in
    pca_data = pd.DataFrame(pca_data, index=data.index)
    pca_data = pd.merge(pca_data, targets, on="sampleid", how="inner")

    # Plot PCA
    ## Create scatterplot

    sns.scatterplot(x=0, y=1, data=pca_data,
                    hue=target_column,
                    alpha=0.3)
    plt.title("PCoA")
    plt.legend(bbox_to_anchor=(1.01, 1),
               loc=2,
               borderaxespad=0.,
               title=target_column)
    ## Format PC labels to include explained variance
    plt.xlabel(f'PC1 ({round(pca.explained_variance_[0], 2)}%)')
    plt.ylabel(f'PC2 ({round(pca.explained_variance_[1], 2)}%)')
    ## Save fig
    plt.savefig("figures\pca\PCA_" + target_column + ".png", bbox_inches='tight')

    # Return loading scores
    ## Get loading scores from both PC's
    pc1_loading_scores = pd.Series(pca.components_[0], index=data.columns)
    pc2_loading_scores = pd.Series(pca.components_[1], index=data.columns)
    ## Sort loading scores by magnitude
    pc1_loading_scores = pc1_loading_scores.abs().sort_values(ascending=False)
    pc2_loading_scores = pc2_loading_scores.abs().sort_values(ascending=False)
    ## Return extracted series
    return pc1_loading_scores, pc2_loading_scores
