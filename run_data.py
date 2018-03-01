from __future__ import division
import networkx as nx
import pandas as pd
import time
import random
import numpy as np
import math
import prepare_data as prep
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    logi = np.zeros(x.shape)
    logi = 1 / (1 + np.exp(-x))
    assert logi.shape == x.shape
    return logi


def log_cost(X, y, w, reg=0):
    """
    Compute the (regularized) cross entropy and the gradient under the logistic regression model
    using data X, targets y, weight vector w (and regularization reg)

    The L2 regulariation is 1/2 reg*|w_{1,d}|^2 i.e. w[0], the bias, is not regularized

    np.log, np.sum, np.choose, np.dot may be usefull here
    Args:
        X: np.array shape (n,d) float - Features
        y: np.array shape (n,)  int - Labels
        w: np.array shape (d,)  float - Initial parameter vector
        reg: scalar - regularization parameter

    Returns:
      cost: scalar the cross entropy cost of logistic regression with data X,y using regularization parameter reg
      grad: np.arrray shape(n,d) gradient of cost at w with regularization value reg
    """
    cost = 0
    grad = np.zeros(w.shape)

    a = sigmoid(X @ w)

    cost = -(y @ np.log(a) + (1 - y) @ np.log(1.0001 - a)) * (1 / len(y)) + (1 / 2) * reg * np.dot(w[1:], w[1:])

    grad = 1 / len(y) * (-X.T @ (y - a)) + reg * np.insert(w[1:], 0, 0)

    ### END CODE
    assert grad.shape == w.shape
    return cost, grad


def mini_batch_grad_descent(X, y, w=None, reg=0, lr=0.1, batch_size=16, epochs=10):
    """
      Run mini-batch stochastic Gradient Descent for logistic regression
      use batchsize data pointsto compute gradient in each step.

    The function np.random.permutation may prove usefull for shuffling the data before each epoch
    It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
    Remeber the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

    Args:
        X: np.array shape (n,d) dtype float32 - Features
        y: np.array shape (n,) dtype int32 - Labels
        w: np.array shape (d,) dtype float32 - Initial parameter vector
        lr: scalar - learning rate for gradient descent
        reg: scalar regularization parameter (optional)
        rounds: rounds to run (optional)
        batch_size: number of elements to use in minibatch
        epochs: Number of scans through the data

    Returns:
        w: numpy array shape (d,) learned weight vector w
    """

    if w is None: w = np.zeros(len(X[1]))
    for i in range(epochs):
        random_batch = np.random.permutation(np.c_[X, y])
        for j in range(int(len(y) / batch_size)):
            cost, grad = log_cost(random_batch[j * batch_size:(j + 1) * batch_size, 0:-1],
                                  random_batch[j * batch_size:(j + 1) * batch_size, -1], w, reg)
            w = w - lr * grad
        lr = lr * 0.92


        # lr = lr * np.linalg.norm(grad)/len(y)

    ### END CODE
    return w


def get_edge_probabilities(feature_matrix, parameters):
    """
    :param feature_matrix:
    :param parameters: beta and alphas in one list
    :return: ingoing edge probabilities for each node
    """
    return sigmoid(feature_matrix @ parameters)


def simulate_activations(G, seeds, edge_probs):

    activations = dict(zip(G.nodes() , [False]*len(G)))
    attempted_activations = np.zeros(len(G))

    new_activations = seeds
    for seed in seeds:
        activations[seed] = True

    while len(new_activations) > 0:
        current_activation = new_activations[0]
        for neighbour in G[current_activation]:
            already_activated = activations[neighbour]
            if not already_activated:
                attempted_activations[int(neighbour)] += 1
                prob = edge_probs[int(neighbour)]
                if random.random() < prob:
                    activations[neighbour] = True
                    new_activations.append(neighbour)

        new_activations.pop(0)

    return activations, attempted_activations


if __name__ == "__main__":
    numberOfFeatures = 3882
    numberOfNodes = 7420

    G = prep.read_graph('vk_mv.txt')
    feature_matrix = prep.get_feature_matrix('vk_mem.txt', numberOfNodes, numberOfFeatures)
    alphas = prep.generate_alphas(numberOfFeatures)

    beta = -3

    parameters = np.insert(alphas, 0, beta)

    feature_matrix = feature_matrix[:, :200]
    parameters = parameters[:200]

    #DO AVERAGING
    for i in range(feature_matrix.shape[0]):
        rowsum = np.sum(feature_matrix[i, 1:])
        if rowsum > 0:
            feature_matrix[i, 1:] = feature_matrix[i, 1:]/rowsum

    edge_probs = get_edge_probabilities(feature_matrix, parameters)

    training_data = []
    training_labels = []

    for j in range(4):
        seeds = [i*(j+1) for i in range(500)]
        activations, attempted_activations = simulate_activations(G, seeds, edge_probs)
        random.seed(j)

        for node, attempts in enumerate(attempted_activations):
            if attempts > 0:
                training_data.append(feature_matrix[node])
                training_labels.append(activations[node])

            if attempts > 1:
                for _ in range(int(attempts) - 1):
                    training_data.append(feature_matrix[node])
                    training_labels.append(False)

    print(np.array(training_data).shape)

    training_data = np.array(training_data)

    training_labels = np.array(training_labels)

    #print('max number of features: ', max(training_data.sum(axis=1)))
    #print(training_labels)


    print('average edge prob was: ', sum(edge_probs)/numberOfNodes)

    #print('total activations were: ', sum(activations.values()))

    #print(parameters.shape)
    #print(training_data.shape)

    logisticRegr = LogisticRegression(fit_intercept=False)
    logisticRegr.fit(training_data, training_labels)
    print('actual params: ', parameters)
    print('predicted params: ', logisticRegr.coef_)
    print('actual probs: ', edge_probs)
    predictedProbs = get_edge_probabilities(feature_matrix, logisticRegr.coef_[0])
    print('predicted probs: ', predictedProbs)
    print('On average parameter estimate is this far off: ', np.mean(np.abs(parameters-logisticRegr.coef_)))
    print('On average probability estimate was this far off: ', np.mean(np.abs(edge_probs-predictedProbs)))
    #print(logisticRegr.intercept_)

    #print((logisticRegr.predict_proba(training_data)).mean())
    #print(mini_batch_grad_descent(training_data, training_labels))