import numpy as np
import pandas as pd
import sklearn
import torch
from joblib import Parallel, delayed
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from torch import nn as nn
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_pca(x, n_components=None):
    total_pca = decomposition.PCA()
    total_pca.fit(np.squeeze(x.reshape(x.shape[0], -1)))
    # Compute cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(total_pca.explained_variance_ratio_)
    if n_components is not None and n_components < 1.0:
        n_components = np.argmax(cumulative_variance_ratio >= n_components) + 1
    else:
        n_components = n_components or len(cumulative_variance_ratio)

    # do not allow for too few components if the data is larger than 100 features
    if len(cumulative_variance_ratio) > 100:
        n_components = max(n_components, 8)

    n_components = int(n_components)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(np.squeeze(x.reshape(x.shape[0], -1)))
    return total_pca, pca


def train_and_evaluate(clf, X_train, X_test, y_train, y_test, name, method):
    clf = sklearn.clone(clf)

    if y_train.ndim > 1 and y_train.shape[1] > 1:
        clf = MultiOutputClassifier(clf)

    clf.fit(X_train, y_train)

    results = []
    for X, y, split in zip([X_train, X_test], [y_train, y_test], ["Train", "Test"]):
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=np.nan)
        results.append({
                "Classifier": name,
                "Split": split,
                "Transformation": method,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
        })
    return results


def run_classification_pipeline(X_train_pca, X_test_pca, X_train_polca, X_test_polca, y_train, y_test):
    classifiers = {
            "Logistic Regression": LogisticRegression(solver="saga"),
            # "Gaussian Naive Bayes": GaussianNB(),
            "Linear SVM": SVC(kernel="linear"),
            "Ridge Classifier": RidgeClassifier(),
            "Perceptron": Perceptron(),
    }

    tasks = [
            (clf, X_train, X_test, name, method)
            for name, clf in classifiers.items()
            for (X_train, X_test, method) in [
                    (X_train_pca, X_test_pca, "PCA"),
                    (X_train_polca, X_test_polca, "POLCA")
            ]
    ]

    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate)(clf, X_train, X_test, y_train, y_test, name, method)
        for clf, X_train, X_test, name, method in tasks
    )

    return pd.DataFrame([item for sublist in results for item in sublist])


class TorchPCA(nn.Module):
    def __init__(self, n_components=None, center=True, device=None):
        """
        PCA implementation using PyTorch, with NumPy interface.

        Parameters:
        - n_components (int or None): Number of components to keep.
                                      If None, all components are kept.
        - center (bool): Whether to center the data before applying PCA.
        - device (str or torch.device or None): The device to run computations on. If None, defaults to 'cpu'.
        """
        super(TorchPCA, self).__init__()
        self.n_components = n_components
        self.center = center
        self.device = device if device is not None else torch.device('cpu')
        self.mean = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the PCA model to X.

        Parameters:
        - X (np.ndarray): The data to fit, of shape (n_samples, n_features).
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.center:
            self.mean = X.mean(dim=0)
            X = X - self.mean

        # Perform PCA using torch.pca_lowrank
        U, S, V = torch.pca_lowrank(X, q=self.n_components)

        self.components_ = V  # Principal components
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ = explained_variance
        total_variance = explained_variance.sum()
        self.explained_variance_ratio_ = explained_variance / total_variance

        # Convert back to numpy arrays for the external interface
        self.mean = self.mean.cpu().numpy() if self.mean is not None else None
        self.components_ = self.components_.cpu().numpy()
        self.explained_variance_ = self.explained_variance_.cpu().numpy()
        self.explained_variance_ratio_ = self.explained_variance_ratio_.cpu().numpy()

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        Parameters:
        - X (np.ndarray): New data to project, of shape (n_samples, n_features).

        Returns:
        - X_transformed (np.ndarray): The data in the principal component space.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.center and self.mean is not None:
            X = X - torch.tensor(self.mean, dtype=torch.float32).to(self.device)
        X_transformed = torch.matmul(X, torch.tensor(self.components_, dtype=torch.float32).to(self.device))
        return X_transformed.cpu().numpy()

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        - X (np.ndarray): The data to fit and transform, of shape (n_samples, n_features).

        Returns:
        - X_transformed (np.ndarray): The data in the principal component space.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Parameters:
        - X_transformed (np.ndarray): Data in the principal component space, of shape (n_samples, n_components).

        Returns:
        - X_original (np.ndarray): The data transformed back to the original space.
        """
        X_transformed = torch.tensor(X_transformed, dtype=torch.float32).to(self.device)
        X_original = torch.matmul(X_transformed,
                                  torch.tensor(self.components_, dtype=torch.float32).t().to(self.device))
        if self.center and self.mean is not None:
            X_original = X_original + torch.tensor(self.mean, dtype=torch.float32).to(self.device)
        return X_original.cpu().numpy()


class ReducedPCA:
    def __init__(self, pca, n_components):
        """
        Initialize ReducedPCA with a fitted PCA object and desired number of components.

        :param pca: Fitted PCA object
        :param n_components: Number of components to use (must be <= pca.n_components_)
        """
        if n_components > pca.n_components_:
            raise ValueError("n_components must be <= pca.n_components_")

        self.n_components = n_components
        self.components_ = pca.components_[:n_components]
        self.mean_ = pca.mean_
        self.explained_variance_ = pca.explained_variance_[:n_components]
        self.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]

    def transform(self, X):
        """
        Apply dimensionality reduction to X using the reduced number of components.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array-like of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space using the reduced number of components.

        :param X_transformed: Array-like of shape (n_samples, n_components)
        :return: Array-like of shape (n_samples, n_features)
        """
        return np.dot(X_transformed, self.components_) + self.mean_
