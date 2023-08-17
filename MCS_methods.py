from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate


def read_prepare_data(rng):
    df = pd.read_csv("Data/ILPD.csv")
    df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)
    df['alkphos'].fillna(df['alkphos'].mean(), inplace=True)
    dataset = df.to_numpy()
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    sm = SMOTENC(categorical_features=[0, 1], random_state=rng)
    X, y = sm.fit_resample(X, y)
    return X, y


def print_anova_fsocres(X, y):
    fs = SelectKBest(score_func=f_classif, k='all')
    headers = ["age", "gender", "tot_bilirubin", "direct_bilirubin", "tot_proteins", "albumin", "ag_ratio", "sgpt",
               "sgot", "alkphos"]
    fs.fit(X, y)
    scores_after_resampling = np.expand_dims(fs.scores_, axis=0)
    scores_after_resampling_table = tabulate(scores_after_resampling, headers, floatfmt=".4f")
    print("\nAnova F-score after resampling:\n", scores_after_resampling_table)


def overall_local_accuracy(classifiers, X_val, y_val, X_test, k=7):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_val, y_val)
    y_pred = []

    for x in X_test:
        # Find the k nearest neighbors of the instance in the training set
        neighbors = knn.kneighbors([x], return_distance=False)
        local_accuracies = []

        # Calculate the local accuracy for each classifier
        # on the k nearest neighbors
        for clf in classifiers:
            local_predictions = clf.predict(X_val[neighbors.ravel()])
            local_accuracies.append(accuracy_score(y_val[neighbors.ravel()], local_predictions))

        # Select the classifier with the highest local accuracy
        best_classifier_idx = np.argmax(local_accuracies)
        y_pred.append(classifiers[best_classifier_idx].predict([x])[0])

    return y_pred


def knorae_predict(trained_classifiers, X_train, y_train, X_test, k=7):
    # Fit the k-NN model to the training data
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predictions for each test sample
    y_pred = []

    for x_test in X_test:
        # Find the indices of k-nearest neighbors for the current test sample
        _, neighbors_indices = knn.kneighbors(x_test.reshape(1, -1))
        neighbors_indices = neighbors_indices.flatten()

        # Oracle classifier indices
        oracle_indices = []

        # Keep reducing the neighbors until at least one oracle classifier is found
        while len(neighbors_indices) > 0:
            # Calculate the accuracy of each classifier within the current neighbors
            accuracy = np.zeros(len(trained_classifiers))
            for i, classifier in enumerate(trained_classifiers):
                accuracy[i] = np.mean(y_train[neighbors_indices] == classifier.predict(X_train[neighbors_indices]))

            # Find the indices of the classifiers with 100% accuracy (oracle classifiers)
            oracle_indices = np.argwhere(accuracy == 1.0).flatten()

            # Break the loop if at least one oracle classifier is found
            if oracle_indices.size > 0:
                break

            # Remove the farthest neighbor and continue the loop
            neighbors_indices = neighbors_indices[:-1]

        # If no oracle classifiers are found, use all classifiers
        if len(oracle_indices) == 0:
            oracle_indices = np.arange(len(trained_classifiers))

        # Obtain the predictions from the oracle classifiers and extract the single prediction value
        oracle_votes = [classifier.predict(x_test.reshape(1, -1))[0] for i, classifier in enumerate(trained_classifiers)
                        if i in oracle_indices]

        # Find the most common prediction (majority vote) among the oracle classifiers
        most_common_vote = Counter(oracle_votes).most_common(1)[0][0]

        y_pred.append(most_common_vote)

    # Return the final predictions as a NumPy array
    return np.array(y_pred)


def des_p_predict(trained_classifiers, X_train, y_train, X_test, k=7):
    # Initialize a K-Nearest Neighbors (k-NN) classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the k-NN classifier to the training data
    knn.fit(X_train, y_train)

    # Initialize an empty list to store the predictions for each test sample
    y_pred = []

    # Set the performance threshold for selecting classifiers
    random_classifier_performance = 0.5

    # Iterate through each test sample
    for x_test in X_test:
        # Find the indices of the k-nearest neighbors for the current test sample
        _, neighbors_indices = knn.kneighbors(x_test.reshape(1, -1))
        neighbors_indices = neighbors_indices.flatten()

        # Initialize an empty list to store the oracle classifier indices
        oracle_indices = []

        # Calculate the accuracy of each classifier within the neighbors
        for i, classifier in enumerate(trained_classifiers):
            accuracy = np.mean(y_train[neighbors_indices] == classifier.predict(X_train[neighbors_indices]))

            # Check if the classifier's performance exceeds the performance threshold
            if accuracy > random_classifier_performance:
                oracle_indices.append(i)

        # If no oracle classifiers are found, use all classifiers
        if len(oracle_indices) == 0:
            oracle_indices = list(range(len(trained_classifiers)))

        # Obtain the predictions from the oracle classifiers
        oracle_votes = [classifier.predict(x_test.reshape(1, -1))[0] for i, classifier in enumerate(trained_classifiers) if i in oracle_indices]

        # Find the most common prediction (majority vote) among the oracle classifiers
        most_common_vote = Counter(oracle_votes).most_common(1)[0][0]

        # Append the most common prediction to the list of predictions
        y_pred.append(most_common_vote)

    # Return the final predictions as a NumPy array
    return np.array(y_pred)