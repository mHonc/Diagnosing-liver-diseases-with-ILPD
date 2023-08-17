from sklearn import clone
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler

from MCS_methods import*
from PoolMethods import*


def main():
    rng = np.random.RandomState(42)
    X, y = read_prepare_data(rng)

    n_splits = 5
    n_repeats = 2
    n_total_features = len(X[0])
    feature_number = 10
    min_feature_number = 5
    scores = np.zeros((4, 6, n_splits*n_repeats))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rng)
    pool_size = 50

    # pick one from list below
    #classifier_to_test = MLPClassifier(max_iter=2000)
    #classifier_to_test = GaussianNB()
    # classifier_to_test = KNeighborsClassifier()
    # classifier_to_test = SVC()
    # classifier_to_test = LogisticRegression(max_iter=2000)
    # classifier_to_test = MLPClassifier(max_iter=2000) # 1 hidden layer, 20 nodes/neurons in this layer
    classifier_to_test = DecisionTreeClassifier()

    # main loop
    X_selected = SelectKBest(score_func=f_classif, k=feature_number).fit_transform(X, y)
    #X_selected = X
    for n_features in range(min_feature_number, n_total_features + 1):
        for fold_id, (train_val_index, test_index) in enumerate(rkf.split(X, y)):
            # Split the train_val data into a smaller training set and a validation set
            X_train_val, X_test = X_selected[train_val_index], X_selected[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            # scaler = StandardScaler()
            # X_train_val = scaler.fit_transform(X_train_val)
            # X_test = scaler.transform(X_test)

            # split data into traning and validation set
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=rng)

            # single classifier
            clf = clone(classifier_to_test)
            clf.fit(X_train_val, y_train_val)
            single_predict = clf.predict(X_test)
            accuracy_single = accuracy_score(y_test, single_predict)
            print(f"Single accuracy: {accuracy_single:.4f}")
            scores[0, n_features - min_feature_number, fold_id] = accuracy_single

            # homogeneus pool
            pool_classifiers = create_homogeneus_classifiers_pool(classifier_to_test, pool_size)
            pool_classifiers = train_homogeneus_classifiers_pool(pool_classifiers, X_train, y_train)

            # Apply the OLA method
            y_pred = overall_local_accuracy(pool_classifiers, X_val, y_val, X_test, k=7)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"OLA accuracy: {accuracy:.4f}")
            scores[1, n_features - min_feature_number, fold_id] = accuracy

            # Apply the KNORAE method
            y_pred = knorae_predict(pool_classifiers, X_val, y_val, X_test, k=7)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"KNORAE accuracy: {accuracy:.4f}")
            scores[2, n_features - min_feature_number, fold_id] = accuracy

            # Apply the DESP method
            y_pred = des_p_predict(pool_classifiers, X_val, y_val, X_test, k=7)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"DESP accuracy: {accuracy:.4f}")
            scores[3, n_features - min_feature_number, fold_id] = accuracy
            print("")

    # Print scores table
    mean_scores = np.mean(scores, axis=2)
    headers = [str(i) for i in range(min_feature_number, n_total_features+1)]
    rows = np.array([["Base"], ["OLA"], ["Knorae"], ["DESP"]])
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

