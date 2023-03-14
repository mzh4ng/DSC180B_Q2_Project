# experimental_strategy          - ohe
# analyte_amount                 - numerical (normalize?)
# analyte_A260A280Ratio          - numerical (contains zero?)
# aliquot_concentration          - numerical (omit 2.10 value)
# gender                         - ohe (55 nan)
# race                           - ohe (1462 nan, keep)
# ethnicity                      - ohe (3096 nan, keep)
# disease_type                   - ohe
# sample_type                    - ordinal [['Primary Tumor', 'Recurrent Tumor', 'Additional - New Primary', 'Metastatic', 'Blood Derived Normal', 'Solid Tissue Normal']]
# primary_site                   - ohe
# age_at_diagnosis               - numerical
# reference_genome               - ohe
# data_submitting_center_label   - ohe
# investigation                  - ohe
# days_to_death                  - numerical
# tissue_source_site_label       - ordinal (avoid ohe, 179 unique vals)
# country_of_sample_procurement  - ohe
# pathologic_t_label             - ohe (reduce stages?)
# pathologic_n_label             - ohe (reduce stages?)
# histological_diagnosis_label   - ordinal (avoid ohe, 71 unique vals)
# pathologic_stage_label         - ohe (reduce stages)
# PlateCenter                    - numerical
# PlateCenterFlag                - ohe

import numpy as np
import pandas as pd
from src.preprocessing import data_cleaning
from src.preprocessing import build_features

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

###
ohe_feats = ['experimental_strategy', 'gender', 'race', 'ethnicity', 'disease_type', 'primary_site',
             'reference_genome', 'data_submitting_center_label', 'investigation', 'country_of_sample_procurement',
             'pathologic_n_label', 'PlateCenterFlag', 'sample_type']
ordinal_feats = ['tissue_source_site_label', 'histological_diagnosis_label']
scaler_feats = ['analyte_amount', 'analyte_A260A280Ratio', 'aliquot_concentration']
passthrough_feats = ['age_at_diagnosis', 'days_to_death']
drop_feats = ['sample_name', 'run_prefix', 'cgc_base_name',
              'filename', 'cgc_id', 'cgc_filename', 'vital_status',
              'data_subtype', 'tcga_sample_id', 'cgc_case_uuid', 'cgc_platform',
              'gdc_file_uuid', 'cgc_sample_uuid',
              'cgc_aliquot_uuid', 'tcga_aliquot_id',
              'tcga_case_id', 'knightlabID', 'portion_is_ffpe', 'pathologic_t_label', 'PlateCenter']
###

def preprocess_metadata(config, df):
    # drop Y col
    df = df.drop(config["dataset"]["y_col"], axis=1)

    # Fill np.nan in ordinal columns (OrdinalEncoder doesn't work otherwise)
    df[config["preprocessing"]["ordinal_cols"]] = df[config["preprocessing"]["ordinal_cols"]].fillna('None')

    ct = make_column_transformer(
        (OneHotEncoder(sparse_output=False), config["preprocessing"]["one_hot_encode_cols"]),
        (OrdinalEncoder(), config["preprocessing"]["ordinal_cols"]),
        (StandardScaler(), config["preprocessing"]["scaler_cols"]),
        ("passthrough", config["preprocessing"]["passthrough_cols"]),
        ("drop", config["preprocessing"]["drop_cols"]),
    )

    transformed = ct.fit_transform(df)

    column_names = (
            ct.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
            + config["preprocessing"]["ordinal_cols"]
            + config["preprocessing"]["scaler_cols"]
            + config["preprocessing"]["passthrough_cols"]
    )

    imputer = SimpleImputer(missing_values=np.nan, strategy=config["preprocessing"]["impute_strat"])
    imputed = imputer.fit_transform(transformed)
    
    return pd.DataFrame(imputed, columns=column_names, index=df.index)


def preprocess(config, raw_metadata, counts):
    metadata = raw_metadata.replace('Not available', np.nan)
    for col in config["preprocessing"]["reduce_cancer_stage_cols"]:
        metadata[col] = data_cleaning.reduce_stages(metadata[col])
    Y = metadata[config["dataset"]["y_col"]]

    # perform experiment specific preprocessing
    if config["experiment_type"] == "classification":
        # filter cancer stage column s.t. only stages I, II, III, and IV remain
        metadata = metadata[metadata[config["dataset"]["y_col"]].isin(["Stage I", "Stage II", "Stage III", "Stage IV"])]
        Y = Y.loc[metadata.index]
        if config["preprocessing"]["y_col_OHE"]:
            Y = build_features.OHE_col(Y)
    if config["experiment_type"] == "regression":
        # constrain range by dropping outliers and NaNs from Y
        metadata = data_cleaning.filter_metadata(metadata)
        Y = Y.loc[metadata.index]

    # preprocess metadata and combine with counts
    counts = counts.loc[metadata.index]
    md = preprocess_metadata(config, metadata)
    X = pd.merge(md, counts, on=config["dataset"]["counts_id_col"], how="inner")

    return X, Y
