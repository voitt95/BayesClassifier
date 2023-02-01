import numpy as np

from .base import NaiveBayes


class GaussianNB(NaiveBayes):
    ''' Implementation of Gasussian Naive Bayes classificator.

    Attributes:
        prior_prob (np.ndarray):  Matrix of shape (n_classes, 2), contains class id and it's a priori probability.
        conditional_means (np.ndarray):  Matrix of shape (n_classes, n_features), contains mean value of every feature for every class.
        conditional_stds (np.ndarray):  Matrix of shape (n_classes, n_features), contains standard deviation of every feature for every class.
        classes (np.ndarray): Vector of shape (n_classes,), contains every class id in vector y.
    '''
    def __init__(self):
        self.prior_prob = {}
        self.conditional_means = {}
        self.conditional_stds = {}

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        '''Fit Gaussian Naive Bayes according to X,y.
        
        Args:
            X (np.ndarray): Matrix of shape (n_samples,n_features), contains `n_samples` of training vectors,
                each vector has `n_features`.
            y (np.ndarray): Vector of shape(n_samples,), contains target values.

        Returns:
            self (GaussianNB): Returns the instance itself.
        '''
        # 1 fill self.class_distrib based on y
        self.X = X
        self.y = y
        self.classes, y_counts =np.unique(self.y,return_counts=True)
        y_counts = y_counts / self.y.shape[0]
        self.y_counts= np.log(y_counts)
        self.prior_prob = np.stack((self.classes,y_counts))

        # 2. Calculate P(X|c) for c in y
        # fill self.conditional_stds based on X and y
        # fill self.conditional_means based on X and y
        self.conditional_parameters = np.zeros((self.classes.shape[0], self.X.shape[1], 2))
        for i,c in enumerate(self.classes):
            idx = np.where(self.y==c)
            x_c_mean = self.X[idx].mean(axis=0)
            x_c_std = self.X[idx].std(axis=0)
            self.conditional_parameters[i,:,0]= x_c_mean
            self.conditional_parameters[i,:,1]= x_c_std
        self.conditional_means = self.conditional_parameters[:,:,0]
        self.conditional_stds = self.conditional_parameters[:,:,1]
        
        return self
        
    @staticmethod
    def _calculate_loglikelihood(mean: np.ndarray, stds: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ Gaussian loglikelihood of the data x given mean and stds.

        Args:
            means (np.ndarray):  Matrix of shape (n_classes, n_features), contains mean value of every feature for every class.
            stds (np.ndarray):  Matrix of shape (n_classes, n_features), contains standard deviation of every feature for every class.
            x (np.ndarray): Vector of shape (n_features,), with data to assign it's loglikelihood.
        
        Returns:
            loglikelihood (np.ndarray):  Matrix of shape (n_classes, n_features), contains logliklihoodes for vector x.
        """
        eps = 1e-4 # Added in denominator to prevent division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * stds + eps)
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * stds + eps)))
        log_likelihood = np.log(coeff * exponent)
        return log_likelihood

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        '''Perform classification on an array of test vectors X.

        Args:
            X (np.ndarray): Matrix of shape (n_samples,n_features), contains test vectors.
        
        Returns:
            y_hat (np.ndarray): Vector of shape(n_samples), contains predicted class id for test vectors/
        '''
        y_hat = np.zeros((X_test.shape[0],1))
        for  idx, x in enumerate(X_test):
            log_likelihood= self._calculate_loglikelihood(self.conditional_means, self.conditional_stds, x).sum(axis=1)
            log_map = log_likelihood + self.y_counts
            y_hat[idx] = self.classes[np.argmax(log_map)]
        return y_hat
