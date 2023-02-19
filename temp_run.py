import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.dataset import make_dataset
from src.preprocessing import build_features
from src.preprocessing import data_cleaning
from src.preprocessing import preprocessing
from src.models import train_model
from src.visualizations import visualize

# set random state
rand_state = 1

# Load files into memory
counts_filename = 'data/count_data_species_raw_WIS_overlapping_fungi_bacteria_12773samples.tsv'
metadata_filename = 'data/metadata_species_WIS_overlapping_fungi_bacteria_12773samples.tsv'

counts = make_dataset.read_fungi_data(counts_filename)
metadata = make_dataset.read_fungi_data(metadata_filename)

# Data cleaning 
metadata = metadata.replace('Not available', np.nan)

metadata = data_cleaning.filter_metadata(metadata)

metadata['pathologic_t_label'] = data_cleaning.reduce_stages(metadata['pathologic_t_label'])
metadata['pathologic_n_label'] = data_cleaning.reduce_stages(metadata['pathologic_n_label'])
metadata['pathologic_stage_label'] = data_cleaning.reduce_stages(metadata['pathologic_stage_label'])

# Preprocessing

metadata = preprocessing.preprocess_metadata(metadata)

metadata.to_csv('temp_metadata.csv')