#!/usr/bin/env python

import numpy as np
import pandas as pd
from data_loader import load_simulated_data, load_breast_cancer_data


def mean_negative_loglikelihood(Y, pYhat):
    """
    Function for computing the mean negative loglikelihood.

    Y is a vector of the true 0/1 labels.
    pYhat is a vector of estimated probabilities, where each entry i is p(Y_i=1 | ... )
    """

    return -np.mean(Y*np.log(pYhat) + (1-Y)*np.log(1-pYhat))


def accuracy(Y, Yhat):
    """
    Function for computing accuracy.

    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    return np.sum(Y==Yhat)/len(Y)

def sigmoid(V):
    """
    Function for mapping a vector of floats to probabilities via the sigmoid function
    """

    return 1/(1+np.exp(-V))


class LogisticRegression:

    def __init__(self, learning_rate=0.1, lamda=None):
        """
        Constructor for the class. Learning rate is
        any positive number controlling step size of gradient descent.
        Lamda is a positive number controlling the strength of regularization.
        When None, no penalty is added.
        """

        self.learning_rate = learning_rate
        self.lamda = lamda
        self.theta = None # theta is initialized once we fit the model

    def _calculate_gradient(self, Xmat, Y, theta_p, h=1e-5):
        """
        Helper function for computing the gradient at a point theta_p.
        """
        # get dimensions of the matrix
        n, d = Xmat.shape

        grad_vec = np.zeros(d)
        # TODO: implement gradient calculation code here
        loss_original = mean_negative_loglikelihood(Y, sigmoid(Xmat @ theta_p))
        for i in range(d):
            theta_p_tmp = theta_p.copy()
            theta_p_tmp[i] += h

            loss_plus_h = mean_negative_loglikelihood(Y, sigmoid(Xmat @ theta_p_tmp))

            grad_vec[i] = (loss_plus_h  - loss_original) / h

        # Add L2 regularization if lambda is specified
        if self.lamda is not None and self.lamda > 0:
            grad_vec += 2 * self.lamda * theta_p  # L2 regularization term


        return grad_vec

    def fit(self, Xmat, Y, max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Fit a logistic regression model using training data Xmat and Y.
        """

        # add a column of ones for the intercept
        n, d = Xmat.shape

        # initialize theta and theta new randomly
        theta = np.random.uniform(-1, 1, d)
        theta_new = np.random.uniform(-1, 1, d)
        iteration = 0

        # keep going until convergence
        while iteration < max_iterations and np.mean(np.abs(theta_new-theta)) >= tolerance:

            if verbose:
                print("Iteration", iteration, "theta=", theta)

            theta = theta_new.copy()

            gradient = self._calculate_gradient(Xmat, Y, theta)

            theta_new = theta - self.learning_rate * gradient


            # TODO: Implement gradient descent
            iteration += 1

        self.theta = theta_new.copy()


    def predict(self, Xmat):
        """
        Predict 0/1 labels for a data matrix Xmat based on the following rule:
        if p(Y=1|X) > 0.5 output a label of 1, else output a label of 0
        """

        return (sigmoid(Xmat @ self.theta) > 0.5).astype(int)


def main():
    """
    Edit only the one line marked as # EDIT ME in this function. The rest is used for grading purposes
    """


    #################
    # Simulated data
    #################
    np.random.seed(333)
    Xmat, Y, feature_names = load_simulated_data()
    model = LogisticRegression(learning_rate=0.2)
    model.fit(Xmat, Y, max_iterations=10000)
    theta = [float(round(p, 2)) for p in model.theta]
    Yhat = model.predict(Xmat)
    print("Simulated data results:\n" + "-"*4)
    print("Simulated data fitted weights", {feature_names[i]: theta[i] for i in range(len(feature_names))})

    print("Accuracy", accuracy(Y, Yhat))

    #####################
    # Breast cancer data
    #####################
    feature_names, data = load_breast_cancer_data()
    model_base = LogisticRegression(learning_rate=0.2, lamda=0.0)
    model_base.fit(data["Xmat_train"], data["Y_train"])
    model_lowl2 = LogisticRegression(learning_rate=0.2, lamda=0.01)
    model_lowl2.fit(data["Xmat_train"], data["Y_train"])
    model_highl2 = LogisticRegression(learning_rate=0.2, lamda=0.2)
    model_highl2.fit(data["Xmat_train"], data["Y_train"])

    Yhat_val_base = model_base.predict(data["Xmat_val"])
    Yhat_val_lowl2 = model_lowl2.predict(data["Xmat_val"])
    Yhat_val_highl2 = model_highl2.predict(data["Xmat_val"])

    accuracy_base = accuracy(data["Y_val"], Yhat_val_base)
    accuracy_lowl2 = accuracy(data["Y_val"], Yhat_val_lowl2)
    accuracy_highl2 = accuracy(data["Y_val"], Yhat_val_highl2)

    print("\nBreast cancer data results:\n" + "-"*4)
    print("Validation accuracy no regularization", accuracy_base)
    print("Validation accuracy lamda=0.01", accuracy_lowl2)
    print("Validation accuracy lamda=0.2", accuracy_highl2)

    # choose best model
    best_model = model_lowl2 # EDIT ME
    Yhat_test = best_model.predict(data["Xmat_test"])
    theta = [float(round(p, 2)) for p in best_model.theta]
    print("Test accuracy", accuracy(data["Y_test"], Yhat_test))
    print("Cancer data weights", {feature_names[i]: theta[i] for i in range(len(feature_names))})


if __name__ == "__main__":
    main()
