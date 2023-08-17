from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tabulate import tabulate
from MCS_methods import*


def create_homogeneus_classifiers_pool():
    classifiers = []
    for x in range(50):
        #clf = GaussianNB()
        #clf = KNeighborsClassifier()
        #clf = SVC()
        #clf = LogisticRegression(max_iter=2000)
        clf = MLPClassifier(max_iter=2000) # 1 hidden layer, 20 nodes/neurons in this layer
        #clf = DecisionTreeClassifier()
        classifiers.append(clf)
    return classifiers

def create_heterogeneus_classifiers_pool():
    classifiers = []
    clf1 = GaussianNB()
    clf2 = KNeighborsClassifier()
    clf3 = SVC()
    clf4 = LogisticRegression(max_iter=2000)
    clf5 = MLPClassifier(max_iter=2000) # 1 hidden layer, 20 nodes/neurons in this layer
    #clf6 = DecisionTreeClassifier()
    classifiers.append(clf1)
    classifiers.append(clf2)
    classifiers.append(clf3)
    classifiers.append(clf4)
    classifiers.append(clf5)
    #classifiers.append(clf6)

    return classifiers

def train_homogeneus_classifiers_pool(classifiers, X_train, y_train):
    # Number of bootstrap samples for each classifier
    n_bootstrap_samples = len(X_train)
    # Initialize the list to store the trained classifiers
    trained_classifiers = []
    # Train each classifier on a non-overlapping training set using bootstrapping
    for clf in classifiers:
        # Create a bootstrap sample
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=n_bootstrap_samples)

        # Train the classifier on the bootstrap sample
        clf.fit(X_train_sample, y_train_sample)

        # Add the trained classifier to the list
        trained_classifiers.append(clf)
    return trained_classifiers

def train_heterogeneus_classifiers_pool(classifiers, X_train, y_train):
    trained_classifiers = []
    for clf in classifiers:
        clf.fit(X_train, y_train)
        trained_classifiers.append(clf)
    return trained_classifiers


def main():
    X, y = read_prepare_data(42)
    n_total_features = len(X[0])
    n_splits = 5
    n_repeats = 2
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    classifiers_to_compare = 7 # 3 or 7
    # From 1 to 10 features, 1 means that we start from 1 to 10
    min_feature_number = 5
    scores = np.zeros((classifiers_to_compare, len(X[0]) - min_feature_number + 1, n_splits*n_repeats))

    # Print anova F-score table
    print_anova_fsocres(X,y)

    #X_selected = X
    # Main loop
    for n_features in range(min_feature_number, n_total_features + 1):
        X_selected = SelectKBest(score_func=f_classif, k=n_features).fit_transform(X, y)
        for fold_id, (train_val_index, test_index) in enumerate(rkf.split(X_selected, y)):
            # Split the dataset into training (30%), validation (20%), and test (50%)
            X_train_val, X_test = X_selected[train_val_index], X_selected[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)

            # Use homogeneus classifier pool
            classifiers = create_homogeneus_classifiers_pool()
            # predict homogeneus
            clf = clone(classifiers[0])
            clf.fit(X_train_val, y_train_val)
            single_predict = clf.predict(X_test)
            accuracy_single = accuracy_score(y_test, single_predict)
            scores[0, n_features - min_feature_number, fold_id] = accuracy_single
            # Create pool of trained classifiers
            trained_classifiers = train_homogeneus_classifiers_pool(classifiers, X_train, y_train)

            # # Use heterogeneus classifier pool
            # classifiers = create_heterogeneus_classifiers_pool()
            # # predict heterogeneus
            # for i in range(classifiers_to_compare-2):
            #     clf = clone(classifiers[i])
            #     clf.fit(X_train_val, y_train_val)
            #     single_predict = clf.predict(X_test)
            #     accuracy_single = accuracy_score(y_test, single_predict)
            #     scores[i, n_features - min_feature_number, fold_id] = accuracy_single
            # # Create pool of trained classifiers
            # trained_classifiers = train_heterogeneus_classifiers_pool(classifiers, X_train, y_train)

            # Use OLA method
            y_pred_ola = overall_local_accuracy(trained_classifiers, X_val, y_val, X_test)
            accuracy_ola = accuracy_score(y_test, y_pred_ola)
            print(f"OLA accuracy: {accuracy_ola:.4f}")
            scores[classifiers_to_compare-2, n_features - min_feature_number, fold_id] = accuracy_ola

            # Use knorae method
            y_pred_knorae = knorae_predict(trained_classifiers, X_val, y_val, X_test)
            accuracy_knorae = accuracy_score(y_test, y_pred_knorae)
            print(f"KNORAE accuracy: {accuracy_ola:.4f}")
            scores[classifiers_to_compare-1, n_features - min_feature_number, fold_id] = accuracy_knorae

        # if (n_features == 5):
        #     break

    # Print scores table
    mean_scores = np.mean(scores, axis=2)
    headers = [str(i) for i in range(min_feature_number, n_total_features+1)]
    rows = np.array([["NB"], ["kNN"], ["SVC"], ["LR"], ["MLP"], ["OLA"], ["Knorae"]])
    #rows = np.array([["Single"], ["OLA"], ["Knorae"]])
    score_table = np.concatenate((rows, mean_scores), axis=1)
    score_table = tabulate(score_table, headers, floatfmt=".4f")
    print("\nMean scores:\n", score_table)

    # Print std's table
    std_scores = np.std(scores, axis=2)
    score_table = np.concatenate((rows, std_scores), axis=1)
    score_table = tabulate(score_table, headers, floatfmt=".4f")
    print("\nSTD scores:\n", score_table)



if __name__ == '__main__':
    main()


"""
# Split the dataset into training (30%), validation (20%), and test (50%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=42)
print(len(X_train))
print(len(X_val))
print(len(X_test))

# reapeated fold 5x2
rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
scores = []

for train_val_index, test_index in rkf.split(X):
    # Split the dataset into training (30%), validation (20%), and test (50%)
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=42)
    
    
    
# 5x2 RepeatedKFold
    for n_features in range(1, n_total_features + 1):
    X_selected = SelectKBest(score_func=f_classif, k=n_features).fit_transform(X, y)
    scaler = StandardScaler()
    
    for train_val_index, test_index in rkf.split(X_selected):
        # Split the dataset into training (30%), validation (20%), and test (50%)
        X_train_val, X_test = X_selected[train_val_index], X_selected[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=1)

        # Standardize the data
        scaler.fit(X_train_val)
        X_train_val_scaled = scaler.transform(X_train_val)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Use single classifier
        clf = classifiers[0]
        clf.fit(X_train_val_scaled, y_train_val)
        single_predict = clf.predict(X_test_scaled)
        accuracy_single = accuracy_score(y_test, single_predict)
        scores_single.append(accuracy_single)

        # Create pool of trained classifiers
        trained_classifiers = train_classifiers_pool(classifiers, X_train_scaled, y_train)

        # Use OLA method
        y_pred_ola = overall_local_accuracy(trained_classifiers, X_val_scaled, y_val, X_test_scaled)
        accuracy_ola = accuracy_score(y_test, y_pred_ola)
        scores_ola.append(accuracy_ola)

    mean_score_single = np.mean(scores_single)
    std_score_single = np.std(scores_single)
    mean_score_ola = np.mean(scores_ola)
    std_score_ola = np.std(scores_ola)
    mean_score_knorae = np.mean(scores_knorae)
    std_score_knorae = np.std(scores_knorae)
    print("Single MLP accuracy score: %.5f (%.5f)" % (mean_score_single, std_score_single))
    print("OLA accuracy score: %.5f (%.5f)" % (mean_score_ola, std_score_ola))
    print("Knorae accuracy score: %.5f (%.5f)" % (mean_score_knorae, std_score_knorae))
    
# Train the base classifier using the training set
            clf = clone(classifiers[1])
            scaler = StandardScaler()
            scaler.fit(X_train_val)
            clf.fit(scaler.transform(X_train_val), y_train_val)
            y_pred = clf.predict(scaler.transform(X_test))
            scores_single.append(accuracy_score(y_test, y_pred))

        mean_score_single = np.mean(scores_single)
        std_score_single = np.std(scores_single)
        print("Single MLP accuracy score: %.5f (%.5f)" % (mean_score_single, n_features))
"""