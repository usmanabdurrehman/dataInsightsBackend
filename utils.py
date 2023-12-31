import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR,SVC
from sklearn.feature_selection import RFE,SelectKBest,f_regression,f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from random import randint

from constants import ProblemType, RANDOM_STATE

def find(stack,needle):
	if needle in stack:
		return stack.index(needle)
	else:
		return -1	

def get_problem_type(target):
	unique_vals = target.unique()
	threshold_unique_categories = 12
	if(len(unique_vals)>threshold_unique_categories):
		return ProblemType.REGRESSION
	else:
		return ProblemType.CLASSIFICATION

def get_decomposed_plot_data(X, y):
    colors_points = []
    if get_problem_type(y) == ProblemType.CLASSIFICATION:
        colors = []
        unique_vals_y = y.unique().tolist()
        n = len(unique_vals_y)
        for i in range(n):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        for i in range(len(y)):
            idx = unique_vals_y.index(y[i])
            colors_points.append(colors[idx])
        reduced_X = PCA(n_components=2).fit_transform(X)
        return {
            'problemType': ProblemType.CLASSIFICATION,
            'x': reduced_X[:, 0].tolist(),
            'y': reduced_X[:, 1].tolist(),
            'colorsPoints': colors_points,
            }
    else:
        reduced_X = PCA(n_components=1).fit_transform(X)
        return {
            'problemType': ProblemType.REGRESSION,
            'x': reduced_X.ravel().tolist(),
            'y': y.tolist(),
            'colorsPoints': colors_points,
            }

def get_feature_importances(X, y, number_of_features_to_select):
    if get_problem_type(y) == ProblemType.CLASSIFICATION:
        feature_importances_tree = (
            RandomForestClassifier(random_state=RANDOM_STATE)
            .fit(X, y)
            .feature_importances_
        )
        features_sorted_tree = [
            columns
            for _, columns in sorted(
                zip(feature_importances_tree, X.columns), reverse=True
            )
        ]
        features_selected_RFE = list(
            X.columns[
                RFE(
                    RandomForestClassifier(random_state=RANDOM_STATE),
                    n_features_to_select=number_of_features_to_select,
                )
                .fit(X, y)
                .support_
            ]
        )

        feature_importances_statistical = (
            SelectKBest(score_func=f_classif, k=number_of_features_to_select)
            .fit(X, y)
            .scores_
        )
        features_sorted_statistical = [
            columns
            for _, columns in sorted(
                zip(feature_importances_statistical, X.columns), reverse=True
            )
        ]
        features_sorted_linear = False

    else:
        feature_importances_tree = (
            RandomForestRegressor(random_state=RANDOM_STATE)
            .fit(X, y)
            .feature_importances_
        )
        features_sorted_tree = [
            columns
            for _, columns in sorted(
                zip(feature_importances_tree, X.columns), reverse=True
            )
        ]
        feature_importances_linear = LinearRegression().fit(X, y).coef_

        features_sorted_linear = [
            columns
            for _, columns in sorted(
                zip(feature_importances_linear, X.columns), reverse=True
            )
        ]
        features_selected_RFE = list(
            X.columns[
                RFE(
                    RandomForestRegressor(random_state=RANDOM_STATE),
                    n_features_to_select=number_of_features_to_select,
                )
                .fit(X, y)
                .support_
            ]
        )
        feature_importances_statistical = (
            SelectKBest(score_func=f_regression, k=number_of_features_to_select)
            .fit(X, y)
            .scores_
        )
        features_sorted_statistical = [
            columns
            for _, columns in sorted(
                zip(feature_importances_statistical, X.columns), reverse=True
            )
        ]
    return {
        "linear": features_sorted_linear,
        "tree": features_sorted_tree,
        "statistical": features_sorted_statistical,
    }

def get_best_fit_model(X, y):
    clfs = [
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        RandomForestClassifier(random_state=RANDOM_STATE),
        SVC(),
        KNeighborsClassifier(),
    ]
    regs = [
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        RandomForestRegressor(random_state=RANDOM_STATE),
        SVR(),
        KNeighborsRegressor(),
    ]

    optimized_clfs = []
    optimized_regs = []

    best_model = ""
    scores = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
    idx = 0
    if get_problem_type(y) == ProblemType.CLASSIFICATION:
        for clf in clfs:
            clf.fit(X_train, y_train)
            optimized_clfs.append(clf)
            scores.append(optimized_clfs[idx].score(X_test, y_test))
            idx += 1
        best_model = optimized_clfs[np.argmax(scores)]
    else:
        for reg in regs:
            reg.fit(X_train, y_train)
            optimized_regs.append(reg)
            scores.append(optimized_regs[idx].score(X_test, y_test))
            idx += 1
        best_model = optimized_regs[np.argmax(scores)]
    return {
        "best_model": best_model,
        "model_info": {
            "bestModelName": type(best_model).__name__,
            "accuracy": round((np.max(scores) * 100), 2),
        },
    }
