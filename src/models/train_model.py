import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error


def train_classification(config, X, Y):
    """
    Take in as input the cleaned datasets of the features(X) and the one-hot encoded cancer stages/targets(Y)
           then perform 10-fold validation split and use them to train the model.

           Output: Auroc and Aupr scores of the model

    """
    # cross validation hyperparams
    n_splits = config["model"]["k-folds"]["n_splits"]
    shuffle = config["model"]["k-folds"]["shuffle"]

    # model hyperparams
    loss = config["model"]["loss"]
    learning_rate = config["model"]["lr"]
    n_estimators = config["model"]["n_estimators"]
    max_depth = config["model"]["max_depth"]

    # sets random state for Stratified K-Folds and Gradient Boost Classifier
    skf_random = 0
    clf_random = 1

    skf = StratifiedKFold(n_splits=n_splits, random_state=skf_random, shuffle=shuffle)

    clf = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=clf_random)

    # dict of scores by stages
    total_auroc_data = {}
    total_aupr_data = {}

    for i, stage in enumerate(Y.columns, start=1):
        print("Starting Stage: " + stage)
        y = Y[stage]

        auroc_plt_data = np.array([])
        aupr_plt_data = np.array([])

        for train_index, val_index in skf.split(X, y):
            train_X, train_y = X.iloc[train_index], y.iloc[train_index]
            val_X, val_y = X.iloc[val_index], y.iloc[val_index]

            # re-fit model
            clf.fit(train_X, train_y)

            # predict, probability of positive class predict
            preds = clf.predict_proba(val_X)[:, 1]

            # calculate scores for current model
            auroc = roc_auc_score(val_y, preds)
            aupr = average_precision_score(val_y, preds)

            # record scores for a given stage
            auroc_plt_data = np.append(auroc_plt_data, auroc)
            aupr_plt_data = np.append(aupr_plt_data, aupr)

        # record scores for all stages into a dict
        total_auroc_data[stage] = auroc_plt_data
        total_aupr_data[stage] = aupr_plt_data

    return clf, total_auroc_data, total_aupr_data


def train_regression(config, X, y):
    """
    Take in as input the cleaned datasets of the features(X) and the one-hot encoded cancer stages/targets(Y)
    then perform 10-fold validation split and use them to train the model.

    Output: Auroc and Aupr scores of the model

    """

    # cross validation hyperparams
    n_splits = config["model"]["k-folds"]["n_splits"]
    shuffle = config["model"]["k-folds"]["shuffle"]

    # model hyperparams
    alpha = config["model"]["alpha"]

    # random states of K-Folds
    skf_random = 0

    skf = KFold(n_splits=n_splits, random_state=skf_random, shuffle=shuffle)

    reg = Lasso(alpha)

    scores = {}

    scores["MSE"] = np.array([])

    for train_index, val_index in skf.split(X, y):
        train_X, train_y = X.iloc[train_index], y.iloc[train_index]
        val_X, val_y = X.iloc[val_index], y.iloc[val_index]

        # re-fit model
        reg.fit(train_X, train_y)

        # make predictions
        preds = reg.predict(val_X)

        # record Mean Squared Errors
        mse = mean_squared_error(val_y, preds)
        scores["MSE"] = np.append(scores["MSE"], mse)

    return reg, scores
