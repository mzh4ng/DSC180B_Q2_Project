from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

import numpy as np


## Py script to train and output model ##
def train_classify_cancer_stages(dataset, cancer_stages):
    '''Take in as input the cleaned datasets of the features(X) and the one-hot encoded cancer stages/targets(Y)
       then perform 10-fold validation split and use them to train the model.
       
       Output: Auroc and Aupr scores of the model
       
    '''

    # TODO: define cross validation hyperparams
    n_splits = 10
    skf_random = 0  # DO NOT TOUCH
    shuffle = True
    # TODO: define model hyperparams
    loss = 'exponential'
    learning_rate = 0.1
    n_estimators = 150
    max_depth = 3
    clf_random = 1  # DO NOT TOUCH

    skf = StratifiedKFold(n_splits=n_splits, random_state=skf_random, shuffle=True)

    clf = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=clf_random)

    total_auroc_data = {}  # dict of scores by stages
    total_aupr_data = {}

    for i, stage in enumerate(cancer_stages.columns, start=1):
        print("Starting Cancer Stage: " + stage)
        X = dataset
        y = cancer_stages[stage]

        auroc_plt_data = np.array([])
        aupr_plt_data = np.array([])

        for train_index, val_index in skf.split(X, y):
            train_X, train_y = X.iloc[train_index], y.iloc[train_index]
            val_X, val_y = X.iloc[val_index], y.iloc[val_index]

            clf.fit(train_X, train_y)  # re-fit model

            preds = clf.predict_proba(val_X)[:, 1]  # predict, probability of positive class predict

            auroc = roc_auc_score(val_y, preds)
            aupr = average_precision_score(val_y, preds)

            auroc_plt_data = np.append(auroc_plt_data, auroc)
            aupr_plt_data = np.append(aupr_plt_data, aupr)

        total_auroc_data[stage] = auroc_plt_data
        total_aupr_data[stage] = aupr_plt_data

    return clf, total_auroc_data, total_aupr_data


def train_regression(X, y):
    '''Take in as input the cleaned datasets of the features(X) and the one-hot encoded cancer stages/targets(Y)
       then perform 10-fold validation split and use them to train the model.

       Output: Auroc and Aupr scores of the model

    '''

    # TODO: define cross validation hyperparams
    n_splits = 10
    skf_random = 0  # DO NOT TOUCH
    shuffle = True
    # TODO: define model hyperparams
    alpha = 0.01

    skf = KFold(n_splits=n_splits, random_state=skf_random, shuffle=shuffle)

    reg = Lasso(alpha)

    print("Starting Days to Die Regression . . .")

    mses = np.array([])

    for train_index, val_index in skf.split(X, y):
        train_X, train_y = X.iloc[train_index], y.iloc[train_index]
        val_X, val_y = X.iloc[val_index], y.iloc[val_index]

        reg.fit(train_X, train_y)  # re-fit model

        preds = reg.predict(val_X)  # predict

        mse = mean_squared_error(val_y, preds)
        mses = np.append(mses, mse)

    return reg, mses
