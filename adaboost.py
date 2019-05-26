"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.D_of_last_iteration = None

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = y.size
        # check shape
        D = np.full((m,), 1 / m)
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            prediction = self.h[t].predict(X)
            results = prediction + y
            epsilon = np.sum(np.where(results == 0, 1, 0) * D)
            self.w[t] = 0.5 * np.log(1 / epsilon - 1)
            D = D * np.exp((-self.w[t]) * prediction * y)
            un_normalized_sum = np.sum(D)
            D = D / un_normalized_sum
        self.D_of_last_iteration = D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        vote = np.zeros(X.shape[0])
        for t in range(max_t):
            vote += self.w[t] * self.h[t].predict(X)
        return np.sign(vote)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        results = self.predict(X, max_t)
        return np.argwhere(results + y == 0).size / y.size
