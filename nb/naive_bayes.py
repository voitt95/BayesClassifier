import abc

import numpy as np

# TODO @Waldek split this file into subdirectory:
# model/
# |__ base.py
# |__ gaussian.py
# |__ multinomial.py

class NB(abc.ABC):
    # TODO @Waldek rename it to NaiveBayes
    @abc.abstractmethod
    def fit(self, X, y): 
        # TODO @Waldek add typehints, e.g. def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        # TODO @Waldek add docstrings EVERYWHERE :D
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


class GaussianNB(NB):
    def __init__(self):
        self.prior_prob = {}
        self.conditional_means = {}
        self.conditional_stds = {}

    
    def fit(self, X, y):
        # 1 fill self.class_distrib based on y
        # np.unique(return_counts=True)
        # (optional) convert to logs 
        # TODO @Waldek what (optional) means here? If it can vary, you can add if statement
        self.X = X
        self.y = y
        self.classes, y_counts =np.unique(self.y,return_counts=True)
        self.y_counts= np.log(y_counts/self.y.shape[0])
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
    def _calculate_likelihood(mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        eps = 1e-4 # Added in denominator to prevent division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var + eps)))
        return np.log(coeff * exponent)

    def predict(self, X_test):
        y_hat = np.zeros((X_test.shape[0],1))
        for  idx, x in enumerate(X_test):
            log_likelihood= self._calculate_likelihood(self.conditional_means, self.conditional_stds, x).sum(axis=1)
            log_map = log_likelihood + self.y_counts
            y_hat[idx] = self.classes[np.argmax(log_map)]
        return y_hat


class MultinomialNB(NB):
    def __init__(self, bins=20):
        self.bins = bins
        
        self.prior_prob = {}
        self.conditional_distribs = {}
        self.mins = {}
        self.maxs = {}

    def fit(self, X, y):
        # 1 fill self.class_distrib based on y
        # np.unique(return_counts=True)
        # (optional) convert to logs
        self.X = X
        self.y = y
        self.classes, y_counts =np.unique(self.y,return_counts=True)
        self.y_counts= np.log(y_counts/self.y.shape[0])
        self.prior_prob = np.stack((self.classes,y_counts))

        # 2. Calculate P(X|c) for c in y
        # fill self.conditional_distribs based on X and y according to self.bins
        self.mins = self.X.min(axis=0)
        self.maxs = self.X.max(axis=0)
        self.conditional_distribs = np.zeros((self.classes.shape[0],self.X.shape[1], self.bins))
        for i, c in enumerate(self.classes):
            idx = np.where(self.y==c)
            for p in range(self.X.shape[1]):
                p_c_hist, _ = np.histogram(X[idx,p], range=(self.mins[p],self.maxs[p]), bins=self.bins, density=True)
                self.conditional_distribs[i,p,:] = p_c_hist
        
        return self

    def predict(self, X_test):
        X_test_digitize = np.empty(X_test.shape)
        for p in range(X_test.shape[1]):
            # TODO @Waldek use more meaningful variable name than "p"
            X_test_digitize[:,p] = np.digitize(X_test[:,p], np.linspace(self.mins[p],self.maxs[p],self.bins))
        X_test_digitize= X_test_digitize.astype(int)

        # 2. P(X|c) self.conditional_distribs for c in y.unique()
        # log(P(X|c))
        likelihoods = np.empty((*X_test.shape,len(self.classes)))
        for p in range(X_test.shape[1]):
            for idx,x in enumerate(X_test_digitize):
                likelihoods[idx,p,:] = self.conditional_distribs[:,p,x[p]-1]
        likelihoods[likelihoods==0] = 0.0001
        log_likelihoods = np.log(likelihoods)

        # 3 add all P(X|c) + self.class_distrib  for c in y.unique()
        y_hat = np.zeros((X_test.shape[0],1),dtype=int)
        for id, l in enumerate(log_likelihoods):
            log_map = l.sum(axis=0) + self.y_counts
            y_hat[id] = self.classes[np.argmax(log_map)]
        # 4. y_hat_idx = argmax 
        return y_hat

