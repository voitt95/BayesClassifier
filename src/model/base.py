import abc

import numpy as np


class NaiveBayes(abc.ABC):
    '''"Abstract base class for naive Bayes classificator.'''
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        '''Fit Naive Bayes according to X,y.

        Args:
            X (np.ndarray): Matrix of shape (n_samples,n_features), contains `n_samples` of training vectors,
                each vector has `n_features`.
            y (np.ndarray): Vector of shape(n_samples,), contains target values.
        '''
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Perform classification on an array of test vectors X.

        Args:
            X (np.ndarray): Matrix of shape (n_samples,n_features), contains `n_samples` of testing vectors,
                each vector has `n_features`.
        '''
        pass