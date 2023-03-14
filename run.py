import sys
import json
import numpy as np
import pandas as pd

from src.dataset import make_dataset
from src.preprocessing import data_cleaning
from src.preprocessing import build_features
from src.models import train_model
from src.visualizations import visualize


def pca(args):
    if "test" in args:
        metadata_filename = "data/test/test_metadata.tsv"
        counts_filename = "data/test/test_fungi.tsv"
        print("Using Test Data.")
    else:
        counts_filename = 'data/count_data_species_raw_WIS_overlapping_fungi_bacteria_12773samples.tsv'
        metadata_filename = 'data/metadata_species_WIS_overlapping_fungi_bacteria_12773samples.tsv'

    # load fungi counts and metadata into
    counts = make_dataset.read_data_file(counts_filename)
    raw_metadata = make_dataset.read_data_file(metadata_filename)

    metadata = raw_metadata.replace('Not available', np.nan)
    for col in ['pathologic_t_label', 'pathologic_n_label', 'pathologic_stage_label']:
        metadata[col] = data_cleaning.reduce_stages(metadata[col])

    target_column = sys.argv[2]

    stages = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

    # filter out missing stage data
    metadata = metadata[metadata.pathologic_stage_label.isin(["Stage I", "Stage II", "Stage III", "Stage IV"])]

    # filter to one experimental strategy
    metadata = metadata[metadata.experimental_strategy == 'WGS']

    # order by cancer stage
    metadata['pathologic_stage_label'] = pd.Categorical(metadata['pathologic_stage_label'], categories=stages)
    metadata = metadata.sort_values(by='pathologic_stage_label')

    # remove cancer stage from pca (keep for coloring plot later)
    targets = metadata[[target_column]]
    metadata.drop(target_column, axis=1)

    # preprocess metadata
    metadata = build_features.preprocess_metadata(config, metadata)
    metadata = metadata.iloc[:, :-7]

    # merge counts data to metadata (drop any counts missing from index in metadata)
    data = pd.merge(metadata, counts, on="sampleid", how="left")

    pca = visualize.create_pca(metadata, targets, target_column)

        # print()
        # print("----  PC1 Loading Scores:  ----")
        # print(pca[0][:5])
        # print()
        # print("----  PC2 Loading Scores:  ----")
        # print(pca[0][:5])


def main(config):
    """
    """
    # load data into memory
    counts = make_dataset.read_data_file(config["dataset"]["counts_file_path"])
    metadata = make_dataset.read_data_file(config["dataset"]["metadata_file_path"])

    # preprocess metadata and combine counts and metadata
    if config["preprocessing"]["do_preprocessing"]:
        X, Y = build_features.preprocess(config, metadata, counts)
    else:
        Y = metadata[config["dataset"]["y_col"]]
        X = pd.merge(metadata, counts, on=config["dataset"]["counts_id_col"], how="inner")

    # train model and visualize
    print("Training Model Now . . .")
    if config["experiment_type"] == "classification":
        model, auroc_plt_data, aupr_plt_data = train_model.train_classification(config, X, Y)
        visualize.plot_classification(config, auroc_plt_data, aupr_plt_data)
    if config["experiment_type"] == "regression":
        model, mses = train_model.train_regression(config, X, Y)
        visualize.plot_regression(config, mses)
    print("Task Completed.")


if __name__ == "__main__":
    """
    """
    args = sys.argv[1:]
    config_name = "config/" + args[0]
    with open(config_name, "r") as jsonfile:
        config = json.load(jsonfile)
    main(config)
