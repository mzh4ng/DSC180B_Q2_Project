import sys
import json
import numpy as np
import pandas as pd

from src.dataset import make_dataset
from src.preprocessing import data_cleaning
from src.preprocessing import build_features
from src.models import train_model
from src.visualizations import visualize


def main(config):
    """
    Main function to run the ML pipeline. Pipeline tuned based upon the inputted config file.
    Loads data into memory, and then preprocess it if specified. Then, trains the model and plots the relevant scores.

    Input: dictionary config file
    Output: None
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
    args = sys.argv[1:]
    config_name = "config/" + args[0]
    with open(config_name, "r") as jsonfile:
        config = json.load(jsonfile)
    main(config)
