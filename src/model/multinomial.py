import numpy as np

from .base import NaiveBayes


class MultinomialNB(NaiveBayes):
    ''' Implementation of Multinomial Naive Bayes classificator.

    Params:
        bins (int): Number of intervals into which continuous feauters must be divided.

    Attributes:
        prior_prob (np.ndarray):  Matrix of shape (n_classes, 2), contains class id and it's a priori probability.
        conditional_distribs (np.ndarray):  Matrix of shape (n_classes, n_features, bins), contains histograms of every feature for every class. 
        classes (np.ndarray): Vector of shape (n_classes,), contains every class id in vector y.
    '''
    def __init__(self, bins: int = 20 ):
        self.bins = bins
        
        self.prior_prob = {}
        self.conditional_distribs = {}
        self.mins = {}
        self.maxs = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNB':
        '''Fit Gaussian Naive Bayes according to X,y.
        
        Args:
            X (np.ndarray): Matrix of shape (n_samples,n_features), contains `n_samples` of training vectors,
                each vector has `n_features`.
            y (np.ndarray): Vector of shape(n_samples,), contains target values.

        Returns:
            self (MultinomialNB): Returns the instance itself.
        '''
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Perform classification on an array of test vectors X.

        Args:
            X (np.ndarray): Matrix of shape (n_samples,n_features), contains test vectors.
        
        Returns:
            y_hat (np.ndarray): Vector of shape(n_samples), contains predicted class id for test vectors/
        '''
        X_test_digitize = np.empty(X.shape)
        for p in range(X.shape[1]):
            X_test_digitize[:,p] = np.digitize(X[:,p], np.linspace(self.mins[p],self.maxs[p],self.bins))
        X_test_digitize= X_test_digitize.astype(int)

        # 2. P(X|c) self.conditional_distribs for c in y.unique()
        # log(P(X|c))
        likelihoods = np.empty((*X.shape,len(self.classes)))
        for feature_id in range(X.shape[1]):
            for idx,x in enumerate(X_test_digitize):
                likelihoods[idx,feature_id,:] = self.conditional_distribs[:,feature_id,x[feature_id]-1]
        likelihoods[likelihoods==0] = 0.0001
        log_likelihoods = np.log(likelihoods)

        # 3 add all P(X|c) + self.class_distrib  for c in y.unique()
        # 4. y_hat_idx = argmax
        y_hat = np.zeros((X.shape[0],1),dtype=int)
        for id, l in enumerate(log_likelihoods):
            log_map = l.sum(axis=0) + self.y_counts
            y_hat[id] = self.classes[np.argmax(log_map)]
         
        return y_hat