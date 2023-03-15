import pandas as pd


def read_data_file(path):
    """
    Reads a tsv file into memory.

    Input: string (path to the file)
    Output: pandas Dataframe
    """
    dataset = pd.read_csv(path, sep='\t', header=0, index_col='sampleid')
    return dataset
