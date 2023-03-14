import pandas as pd


def read_fungi_data(path):
    dataset = pd.read_csv(path, sep='\t', header=0, index_col='sampleid')
    return dataset
