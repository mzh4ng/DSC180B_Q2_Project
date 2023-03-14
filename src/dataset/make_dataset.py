import pandas as pd


def read_data_file(path):
    """
    """
    dataset = pd.read_csv(path, sep='\t', header=0, index_col='sampleid')
    return dataset
