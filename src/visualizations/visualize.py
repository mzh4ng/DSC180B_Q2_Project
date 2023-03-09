import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def abbreviate(string, tcga_abbrev):
    abbr = tcga_abbrev.loc[string][0]
    return abbr

def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = np.mean(values)
    stdev = np.std(values)
    confidence_interval = z * stdev / (len(values)**(1/2))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color=color)

    # TODO: Delete this when plot_model_metrics() is rewritten
    plt.savefig('final_figure.png')

    return mean, confidence_interval

def init_visualization(cancer_stages):
    ## INITIALIZE PLOT ##
    fig = plt.figure()
    y_ticks = plt.yticks(np.arange(11)/10)
    x_ticks = plt.xticks(np.arange(1, len(cancer_stages.columns)+1), [stage for stage in cancer_stages.columns])
    plt.autoscale(False)
    title = plt.title('AUROC')

def create_pca(data, targets):
    # Run MDS
    ## 
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import MDS

    distances = pairwise_distances(data, metric='jaccard')
    mds = MDS(dissimilarity='precomputed', random_state=0)
    pca_data = mds.fit_transform(distances)

    # # Run PCA
    # ## Standardize data
    # scaler = StandardScaler() 
    # scaled_data = scaler.fit_transform(data)
    # ## Run sklearn PCA on data
    # pca = PCA(n_components=2)
    # pca_data = pca.fit_transform(scaled_data)
    ## Convert ndarray to DataFrame, Add targets back in
    pca_data = pd.DataFrame(pca_data, index=data.index)
    pca_data = pd.merge(pca_data, targets, on="sampleid", how="inner")

    # Plot PCA
    ## Create scatterplot
    sns.scatterplot(x=0, y=1, data=pca_data, 
                    hue='pathologic_stage_label',
                    alpha=0.25)
    plt.title("PCoA")
    ## Format PC labels to include explained variance
    plt.xlabel(f'PC1 ({round(pca.explained_variance_[0], 2)}%)')
    plt.ylabel(f'PC2 ({round(pca.explained_variance_[1], 2)}%)')
    ## Save fig
    plt.savefig("PCA.png")

    # Return loading scores
    ## Get loading scores from both PC's
    pc1_loading_scores = pd.Series(pca.components_[0], index=data.columns)
    pc2_loading_scores = pd.Series(pca.components_[1], index=data.columns)
    ## Sort loading scores by magnitude
    pc1_loading_scores = pc1_loading_scores.abs().sort_values(ascending=False)
    pc2_loading_scores = pc2_loading_scores.abs().sort_values(ascending=False)
    ## Return extracted series
    return pc1_loading_scores, pc2_loading_scores