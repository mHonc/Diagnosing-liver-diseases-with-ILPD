from sklearn import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.utils import resample


# uncomment base classifier that will be used
def create_homogeneus_classifiers_pool(clf, pool_size):
    classifiers = []
    for x in range(pool_size):
        classifiers.append(clone(clf))
    return classifiers


def create_heterogeneus_classifiers_pool():
    classifiers = []
    clf1 = GaussianNB()
    clf2 = KNeighborsClassifier(n_neighbors=2)
    clf3 = SVC()
    clf4 = LogisticRegression(max_iter=2000)
    clf5 = MLPClassifier(max_iter=2000)  # 1 hidden layer, 20 nodes/neurons in this layer
    clf6 = DecisionTreeClassifier()

    classifiers.append(clf1)
    classifiers.append(clf2)
    classifiers.append(clf3)
    classifiers.append(clf4)
    classifiers.append(clf5)
    classifiers.append(clf6)

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
