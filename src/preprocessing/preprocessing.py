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

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

ohe_feats = [           'experimental_strategy', 'gender', 'race', 'ethnicity', 'disease_type', 'primary_site',
                        'reference_genome', 'data_submitting_center_label', 'investigation', 'country_of_sample_procurement', 
                        'pathologic_t_label', 'pathologic_n_label', 'PlateCenterFlag', 'sample_type']
ordinal_feats = [       'tissue_source_site_label', 'histological_diagnosis_label']
scaler_feats = [        'analyte_amount', 'analyte_A260A280Ratio', 'aliquot_concentration']
passthrough_feats = [   'age_at_diagnosis']
drop_feats = [          'sample_name', 'run_prefix', 'cgc_base_name',
                        'filename', 'cgc_id', 'cgc_filename', 'vital_status',
                        'data_subtype', 'tcga_sample_id', 'cgc_case_uuid', 'cgc_platform',
                        'gdc_file_uuid', 'cgc_sample_uuid',
                        'cgc_aliquot_uuid', 'tcga_aliquot_id',
                        'tcga_case_id', 'days_to_death', 'knightlabID', 'portion_is_ffpe', 'PlateCenter']

def preprocess_metadata(df):
    ct = make_column_transformer(
            (OneHotEncoder(sparse=False), ohe_feats + ordinal_feats),
        (StandardScaler(), scaler_feats),
        ("passthrough", passthrough_feats),
        ("drop", drop_feats),
    )

    transformed = ct.fit_transform(df)

    column_names = (
        scaler_feats
        + passthrough_feats    
        + ct.named_transformers_["onehotencoder"].get_feature_names().tolist()
    )

    return pd.DataFrame(transformed, columns=column_names, index=df.index)