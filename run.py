import sys
import json
import numpy as np
import pandas as pd

from src.dataset import make_dataset
from src.preprocessing import build_features
from src.preprocessing import data_cleaning
from src.preprocessing import preprocessing
from src.models import train_model
from src.visualizations import visualize


def main2(args):
    if "test" in args:
        metadata_filename = "data/test/test_metadata.tsv"
        counts_filename = "data/test/test_fungi.tsv"
        print("Using Test Data.")
    else:
        counts_filename = 'data/count_data_species_raw_WIS_overlapping_fungi_bacteria_12773samples.tsv'
        metadata_filename = 'data/metadata_species_WIS_overlapping_fungi_bacteria_12773samples.tsv'

    # load fungi counts and metadata into
    counts = make_dataset.read_fungi_data(counts_filename)
    raw_metadata = make_dataset.read_fungi_data(metadata_filename)

    metadata = raw_metadata.replace('Not available', np.nan)
    for col in ['pathologic_t_label', 'pathologic_n_label', 'pathologic_stage_label']:
        metadata[col] = data_cleaning.reduce_stages(metadata[col])

    if "cs" in args:
        # name of cancer stage column
        cancer_stage = "pathologic_stage_label"
        # clean cancer stage column s.t. only stages I, II, III, and IV remain
        metadata = metadata[metadata.pathologic_stage_label.isin(["Stage I", "Stage II", "Stage III", "Stage IV"])]
        counts = counts.loc[metadata.index]

        Y = build_features.OHE_col(metadata[cancer_stage])
        X = metadata.drop(cancer_stage, axis=1)
        X = preprocessing.preprocess_metadata(config, X)
        X = pd.merge(X, counts, on="sampleid", how="inner")

        print("Training Model Now . . .")
        model, auroc_plt_data, aupr_plt_data = train_model.train_classification(config, X, Y)

        visualize.init_visualization(Y)
        i = 1
        for stage in auroc_plt_data.keys():
            visualize.plot_confidence_interval(i, auroc_plt_data[stage])
            i += 1

        return model, (auroc_plt_data, aupr_plt_data)

    if "dtd" in args:
        dtd = "days_to_death"

        metadata = data_cleaning.filter_metadata(metadata)
        counts = counts.loc[metadata.index]

        Y = metadata[dtd]
        # X = metadata.drop(cancer_stage, axis=1)
        # X = X.replace(np.nan, "NAN")
        # X = preprocessing.preprocess_metadata(X)
        # X = pd.merge(X, counts, on="sampleid", how="inner")

        print("Training Model Now . . .")
        model, mses = train_model.train_regression(config, counts, Y)

        # TODO: visualize MSEs

        print("Average MSE for Days to Die Regression:" + str(np.mean(mses)))

        return model, mses

    if "pca" in args:
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
        metadata = preprocessing.preprocess_metadata(config, metadata)
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
    # load data into memory
    counts = make_dataset.read_fungi_data(config["dataset"]["counts_file_path"])
    metadata = make_dataset.read_fungi_data(config["dataset"]["metadata_file_path"])

    # preprocess metadata and combine counts and metadata
    if config["preprocessing"]["do_preprocessing"]:
        X, Y = preprocessing.preprocess(config, metadata, counts)
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


if __name__ == "__main__":
    args = sys.argv[1:]
    config_name = "config/" + args[0]
    with open(config_name, "r") as jsonfile:
        config = json.load(jsonfile)
    main(config)
