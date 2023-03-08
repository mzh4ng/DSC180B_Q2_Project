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
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Run PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Convert ndarray to DataFrame, Add targets back in
    pca_data = pd.DataFrame(pca_data, index=data.index)
    pca_data = pd.merge(pca_data, targets, on="sampleid", how="inner")

    # Plot PCA and save
    sns.scatterplot(0, 1, data=pca_data, hue='pathologic_stage_label')
    plt.savefig("PCA.png")

    # Get loading scores from both PC's
    pc1_loading_scores = pd.Series(pca.components_[0], index=data.columns)
    pc2_loading_scores = pd.Series(pca.components_[1], index=data.columns)
    
    # Sort loading scores by magnitude
    pc1_loading_scores = pc1_loading_scores.abs().sort_values(ascending=False)
    pc2_loading_scores = pc2_loading_scores.abs().sort_values(ascending=False)

    return pc1_loading_scores, pc2_loading_scores


# TODO: Rewrite this function to fit our project
# 
# def plot_model_metrics(plot_data_path, disease_types, tcga_abbrev, dataset_names):
#     """Plot the model AUROC/AUPR scores
#     Args:
#         plot_data_path (String): path of the plot data
#         disease_types (DataFrame): One hot encoded disease_type dataframe 
#         tcga_abbrev (DataFrame): Dataframe of TCGA abbreviations
#     """
#     plot_data = pd.read_csv(plot_data_path, index_col='Metric')
#     # initialize viz figure
#     fig, ax = init_visualization(disease_types, tcga_abbrev)

#     colors = ['red','blue','orange']
#     offsets = [-0.3, 0, 0.3]

#     for color, offset, dataset_name in zip(colors, offsets, plot_data.columns):
#         color = color
#         offset = offset
#         dataset_name = dataset_name

#         auroc = eval(plot_data[dataset_name]['AUROC'])
#         aupr = eval(plot_data[dataset_name]['AUPR'])

#         for i, cancer in enumerate(disease_types.columns, start=1):
#             proportion = disease_types[cancer].mean()

#             plot_confidence_interval(i+offset, auroc[cancer], color=color, axes=ax[0]) #AUROC plot
#             plot_confidence_interval(i+offset, aupr[cancer], color=color, axes=ax[1]) #AUPR plot
#             plot_baseline(proportion=proportion, x=i, axes=ax[1]) #AUPR baselines


#     ax[0].hlines(0.5, xmin=0, xmax=10, linestyle ='dashed', color = 'black')
#     custom_lines = [Line2D([0], [0], color='red'),
#                     Line2D([0], [0], color='blue'),
#                     Line2D([0], [0], color='orange')]
    
#     ax[0].legend(labels=dataset_names,
#                handles=custom_lines, loc='center', bbox_to_anchor=(0.5, 1.23), ncol=3)
#     plt.savefig('final_figure.png')
    
#     return