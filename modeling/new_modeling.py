import datetime
import matplotlib.pyplot as plt
import pickle
import polars as pl
import pprint

from sklearn import svm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# Loading the Data
df = pl.read_parquet("dataset_current.parquet")


# Data Regularization
def scale_features(df, features_to_scale):
    scaler = StandardScaler().fit(df[features_to_scale])
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    return df


# Evaluation
def evaluate(truths, predictions, model, accuracy_list):
    model_name = type(model).__name__
    accuracy = accuracy_score(truths, predictions)
    print(f"{model_name} Results")
    print("------------------------")
    print(f" Accuracy: {accuracy:.5%}")
    print(f"Precision: {precision_score(truths, predictions):.5%}")
    print(f"   Recall: {recall_score(truths, predictions):.5%}")
    print(f"       F1: {f1_score(truths, predictions):.5%}")

    accuracy_list.append([model_name, model, accuracy])

    fpr, tpr, _ = roc_curve(truths, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="orange",
        lw=lw,
        label="ROC curve (area = {:.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


# Modeling with All Stats
def modeling_all_stats(df):
    features_to_scale = [
        "pitcher_era_comp",
        "pitcher_win_percentage_comp",
        "pitcher_win_comp",
        "pitcher_losses_comp",
        "pitcher_innings_pitched_comp",
        "pitcher_k_nine_comp",
        "pitcher_bb_nine_comp",
        "pitcher_k_bb_diff_comp",
        "pitcher_whip_comp",
        "pitcher_babip_comp",
        "pitcher_k_bb_ratio_comp",
    ]
    df = scale_features(df, features_to_scale)

    all_stats_accuracies = []
    all_stat_features = [
        "pitcher_era_comp",
        "pitcher_win_percentage_comp",
        "pitcher_win_comp",
        "pitcher_losses_comp",
        "pitcher_innings_pitched_comp",
        "pitcher_k_nine_comp",
        "pitcher_bb_nine_comp",
        "pitcher_k_bb_diff_comp",
        "pitcher_whip_comp",
        "pitcher_babip_comp",
        "pitcher_k_bb_ratio_comp",
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[all_stat_features],
        df["winning_team"],
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": svm.SVC(),
        "Nearest Centroid": NearestCentroid(),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient-Boosted Tree": HistGradientBoostingClassifier(max_iter=100),
    }

    for model_name, model in models.items():
        if model_name == "k-Nearest Neighbors":
            param_grid = {"n_neighbors": [1, 2, 3, 5, 8, 13, 21, 34, 45, 79]}
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_n_neighbors = grid_search.best_params_["n_neighbors"]
            best_knn_classifier = grid_search.best_estimator_
            model = best_knn_classifier
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        evaluate(y_test, predictions, model, all_stats_accuracies)

    return all_stats_accuracies


# Modeling with Old-School Stats
def modeling_old_school(df):
    features_to_scale = [
        "pitcher_era_comp",
        "pitcher_win_percentage_comp",
        "pitcher_win_comp",
        "pitcher_losses_comp",
        "pitcher_innings_pitched_comp",
    ]
    df = scale_features(df, features_to_scale)

    old_school_accuracies = []
    old_school_features = [
        "pitcher_era_comp",
        "pitcher_win_percentage_comp",
        "pitcher_win_comp",
        "pitcher_losses_comp",
        "pitcher_innings_pitched_comp",
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[old_school_features],
        df["winning_team"],
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": svm.SVC(),
        "Nearest Centroid": NearestCentroid(),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient-Boosted Tree": HistGradientBoostingClassifier(max_iter=100),
    }

    for model_name, model in models.items():
        if model_name == "k-Nearest Neighbors":
            param_grid = {"n_neighbors": [1, 2, 3, 5, 8, 13, 21, 34, 45, 79]}
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_n_neighbors = grid_search.best_params_["n_neighbors"]
            best_knn_classifier = grid_search.best_estimator_
            model = best_knn_classifier
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        evaluate(y_test, predictions, model, old_school_accuracies)

    return old_school_accuracies


# Modeling with Modern Stats
def modeling_modern(df):
    features_to_scale = [
        "pitcher_k_nine_comp",
        "pitcher_bb_nine_comp",
        "pitcher_k_bb_diff_comp",
        "pitcher_whip_comp",
        "pitcher_babip_comp",
        "pitcher_k_bb_ratio_comp",
    ]
    df = scale_features(df, features_to_scale)

    modern_accuracies = []
    modern_features = [
        "pitcher_k_nine_comp",
        "pitcher_bb_nine_comp",
        "pitcher_k_bb_diff_comp",
        "pitcher_whip_comp",
        "pitcher_babip_comp",
        "pitcher_k_bb_ratio_comp",
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[modern_features],
        df["winning_team"],
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": svm.SVC(),
        "Nearest Centroid": NearestCentroid(),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient-Boosted Tree": HistGradientBoostingClassifier(max_iter=100),
    }

    for model_name, model in models.items():
        if model_name == "k-Nearest Neighbors":
            param_grid = {"n_neighbors": [1, 2, 3, 5, 8, 13, 21, 34, 45, 79]}
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_n_neighbors = grid_search.best_params_["n_neighbors"]
            best_knn_classifier = grid_search.best_estimator_
            model = best_knn_classifier
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        evaluate(y_test, predictions, model, modern_accuracies)

    return modern_accuracies


# Noting the Best Classifiers
def best_classifiers_accuracies(all_stats_accuracies, old_school_accuracies, modern_accuracies):
    headers = ["Model", "Accuracy"]

    best_all_stats_accuracy_entry = max(all_stats_accuracies, key=lambda x: x[2])
    best_all_stats_classifier_name = best_all_stats_accuracy_entry[0]
    best_all_stats_accuracy_value = best_all_stats_accuracy_entry[2]

    table_data_all_stats = [
        (row[0], f"{row[2]:.2%}") for row in all_stats_accuracies
    ]

    best_old_school_accuracy_entry = max(old_school_accuracies, key=lambda x: x[2])
    best_old_school_classifier_name = best_old_school_accuracy_entry[0]
    best_old_school_accuracy_value = best_old_school_accuracy_entry[2]

    table_data_old_school = [
        (row[0], f"{row[2]:.2%}") for row in old_school_accuracies
    ]

    best_modern_accuracy_entry = max(modern_accuracies, key=lambda x: x[2])
    best_modern_classifier_name = best_modern_accuracy_entry[0]
    best_modern_accuracy_value = best_modern_accuracy_entry[2]

    table_data_modern = [
        (row[0], f"{row[2]:.2%}") for row in modern_accuracies
    ]

    print("Best All-Stats Model:")
    print(tabulate(table_data_all_stats, headers=headers, tablefmt="fancy_grid"))
    print(f"\nBest All-Stats Model: {best_all_stats_classifier_name}")
    print(f"            Accuracy: {best_all_stats_accuracy_value:.2%}")

    print("\nBest Old-School Model:")
    print(tabulate(table_data_old_school, headers=headers, tablefmt="fancy_grid"))
    print(f"\nBest Old-School Model: {best_old_school_classifier_name}")
    print(f"             Accuracy: {best_old_school_accuracy_value:.2%}")

    print("\nBest Modern Model:")
    print(tabulate(table_data_modern, headers=headers, tablefmt="fancy_grid"))
    print(f"\nBest Modern Model: {best_modern_classifier_name}")
    print(f"         Accuracy: {best_modern_accuracy_value:.2%}")


# Exporting the Best Model
def export_best_model(all_stats_accuracies, old_school_accuracies, modern_accuracies):
    now = datetime.datetime.now()

    all_stats_object = (
        max(all_stats_accuracies, key=lambda x: x[2])[1],
        {
            "date created": now,
            "model type": max(all_stats_accuracies, key=lambda x: x[2])[0],
            "parameters used": ", ".join(
                ["pitcher_era_comp", "pitcher_win_percentage_comp", "pitcher_win_comp", "pitcher_losses_comp", "pitcher_innings_pitched_comp", "pitcher_k_nine_comp", "pitcher_bb_nine_comp", "pitcher_k_bb_diff_comp", "pitcher_whip_comp", "pitcher_babip_comp", "pitcher_k_bb_ratio_comp"]
            ),
            "accuracy": max(all_stats_accuracies, key=lambda x: x[2])[2],
            "training set size": X_train.shape[0],
            "testing set size": X_test.shape[0],
        },
    )

    old_school_object = (
        max(old_school_accuracies, key=lambda x: x[2])[1],
        {
            "date created": now,
            "model type": max(old_school_accuracies, key=lambda x: x[2])[0],
            "parameters used": ", ".join(
                ["pitcher_era_comp", "pitcher_win_percentage_comp", "pitcher_win_comp", "pitcher_losses_comp", "pitcher_innings_pitched_comp"]
            ),
            "accuracy": max(old_school_accuracies, key=lambda x: x[2])[2],
            "training set size": X_train.shape[0],
            "testing set size": X_test.shape[0],
        },
    )

    modern_stats_object = (
        max(modern_accuracies, key=lambda x: x[2])[1],
        {
            "date created": now,
            "model type": max(modern_accuracies, key=lambda x: x[2])[0],
            "parameters used": ", ".join(
                ["pitcher_k_nine_comp", "pitcher_bb_nine_comp", "pitcher_k_bb_diff_comp", "pitcher_whip_comp", "pitcher_babip_comp", "pitcher_k_bb_ratio_comp"]
            ),
            "accuracy": max(modern_accuracies, key=lambda x: x[2])[2],
            "training set size": X_train.shape[0],
            "testing set size": X_test.shape[0],
        },
    )

    filename = "model_objects/current_models.pkl"
    models_object = (all_stats_object, old_school_object, modern_stats_object)
    with open(filename, "wb") as file:
        pickle.dump(models_object, file)


if __name__ == "__main__":
    all_stats_accuracies = modeling_all_stats(df)
    old_school_accuracies = modeling_old_school(df)
    modern_accuracies = modeling_modern(df)
    best_classifiers_accuracies(all_stats_accuracies, old_school_accuracies, modern_accuracies)
    export_best_model(all_stats_accuracies, old_school_accuracies, modern_accuracies)
