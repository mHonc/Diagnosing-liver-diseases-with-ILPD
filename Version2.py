from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
# Importing dataset and preprocessing routines
from sklearn.datasets import fetch_openml
# Base classifier models:
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from deslib.dcs import MCB, LCA
# Example of DCS techniques
from deslib.dcs import OLA
from deslib.des import DESP
# Example of DES techniques
from deslib.des import KNORAE
from deslib.des import KNORAU
from MCS_methods import*

def knora_u_predict(trained_classifiers, X_train, y_train, X_test, k=5):
    # Fit the k-NN model to the training data
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Initialize an empty list to store the predictions for each test sample
    y_pred = []

    # Iterate through each test sample
    for x_test in X_test:
        # Find the indices of k-nearest neighbors for the current test sample
        _, neighbors_indices = knn.kneighbors(x_test.reshape(1, -1))
        neighbors_indices = neighbors_indices.flatten()

        # Initialize an empty list to store the oracle classifier indices
        oracle_indices = []

        # Calculate the accuracy of each classifier within the neighbors
        accuracy = np.zeros(len(trained_classifiers))
        for i, classifier in enumerate(trained_classifiers):
            accuracy[i] = np.mean(y_train[neighbors_indices] == classifier.predict(X_train[neighbors_indices]))

        # Find the indices of the classifiers with >0% accuracy (oracle classifiers)
        oracle_indices = np.argwhere(accuracy > 0.0).flatten()

        # If no oracle classifiers are found, use all classifiers
        if len(oracle_indices) == 0:
            oracle_indices = np.arange(len(trained_classifiers))

        # Obtain the predictions from the oracle classifiers and extract the single prediction value
        oracle_votes = [classifier.predict(x_test.reshape(1, -1))[0] for i, classifier in enumerate(trained_classifiers) if i in oracle_indices]

        # Find the most common prediction (majority vote) among the oracle classifiers
        most_common_vote = Counter(oracle_votes).most_common(1)[0][0]

        # Append the most common prediction to the list of predictions
        y_pred.append(most_common_vote)

    # Return the final predictions as a NumPy array
    return np.array(y_pred)

rng = np.random.RandomState(1410)
# data = fetch_openml(name='phoneme', cache=False, as_frame=False)
# X = data.data
# y = data.target

# Read and prepare data
df = pd.read_csv("Data/ILPD.csv")
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)
df['alkphos'].fillna(df['alkphos'].mean(), inplace=True)
dataset = df.to_numpy()
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# split the data into training and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=rng)

# scaler = StandardScaler()
# X_train_val = scaler.fit_transform(X_train_val)
# X_test = scaler.transform(X_test)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train_val, y_train_val,
                                                    test_size=0.33,
                                                    random_state=rng)

print(len(X_train))
print(len(X_dsel))
print(len(X_test))


pool_classifiers = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(max_iter=2000),
    MLPClassifier(max_iter=2000), # 1 hidden layer, 20 nodes/neurons in this layer
    DecisionTreeClassifier()]

# fit each classifier on the training set
for c in pool_classifiers:
    c.fit(X_train, y_train)

# Initializing the techniques
knorau = KNORAU(pool_classifiers)
kne = KNORAE(pool_classifiers)
# DCS techniques
ola = OLA(pool_classifiers)
new = DESP(pool_classifiers)

# Fitting the DS techniques
knorau.fit(X_dsel, y_dsel)
kne.fit(X_dsel, y_dsel)
ola.fit(X_dsel, y_dsel)
new.fit(X_dsel, y_dsel)

# Calculate classification accuracy of each technique
print('Evaluating Deslib techniques:')
print('Classification accuracy of KNORA-U: ', knorau.score(X_test, y_test))
print('Classification accuracy of KNORA-E: ', kne.score(X_test, y_test))
print('Classification accuracy of OLA: ', ola.score(X_test, y_test))
print('Classification accuracy of new: ', new.score(X_test, y_test))

# Apply the OLA method
y_pred = overall_local_accuracy(pool_classifiers, X_dsel, y_dsel, X_test, k=7)
accuracy = accuracy_score(y_test, y_pred)
print(f"OLA accuracy: {accuracy:.4f}")

# Apply the KNORAE method
y_pred = knorae_predict(pool_classifiers, X_dsel, y_dsel, X_test, k=7)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNORAE accuracy: {accuracy:.4f}")

# Apply the DESP method
y_pred = des_p_predict(pool_classifiers, X_dsel, y_dsel, X_test, k=7)
accuracy = accuracy_score(y_test, y_pred)
print(f"DESP accuracy: {accuracy:.4f}")

# Apply the KNORAU method
y_pred = knora_u_predict(pool_classifiers, X_dsel, y_dsel, X_test, k=7)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNORAU accuracy: {accuracy:.4f}")

for e in pool_classifiers:
    yhat = e.predict(X_test)
    score = accuracy_score(y_test, yhat)
    print('>%s: %.3f' % (e.__class__.__name__, score))

