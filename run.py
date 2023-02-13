import sys
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.dataset import make_dataset
from src.preprocessing import build_features
from src.preprocessing import data_cleaning
from src.preprocessing import preprocessing
from src.models import train_model
from src.visualizations import visualize


def main(args):
    if "test" in args:
        metadata_filename = "data/test/test_metadata.tsv"
        counts_filename = "data/test/test_fungi.tsv"
    else:
        counts_filename = 'data/count_data_species_raw_WIS_overlapping_fungi_bacteria_12773samples.tsv'
        metadata_filename = 'data/metadata_species_WIS_overlapping_fungi_bacteria_12773samples.tsv'

    # load fungi counts and metadata into
    counts = make_dataset.read_fungi_data(counts_filename)
    raw_metadata = make_dataset.read_fungi_data(metadata_filename)

    metadata = raw_metadata.replace('Not available', np.nan)

    if "cs" in args:
        # name of cancer stage column
        cancer_stage = "pathologic_stage_label"
        # clean cancer stage column s.t. only stages I, II, III, and IV remain
        metadata[cancer_stage] = data_cleaning.reduce_stages(metadata[cancer_stage])

        Y = build_features.OHE_col(metadata[cancer_stage])
        #X = metadata.drop(cancer_stage, axis=1)
        #X = X.replace(np.nan, "NAN")
        #X = preprocessing.preprocess_metadata(X)
        #X = pd.merge(X, counts, on="sampleid", how="inner")

        model, auroc_plt_data, aupr_plt_data = train_model.train_classify_cancer_stages(counts, Y)

        visualize.init_visualization(Y)
        visualize.plot_confidence_interval(auroc_plt_data.keys(), auroc_plt_data)

if __name__ == "__main__":
    # args:
    # test = use test dataset
    # dtd = days to death regression
    # cs = cancer stage classification
    args = sys.argv[1:]
    main(args)
    #generates graph in final_figure.png

main("cs")