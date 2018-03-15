
import networkx as nx
import pandas as pd
import time
import random
import numpy as np
import math
import prepare_data as prep
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt



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


def get_edge_probabilities(feature_mat, parameters):
    """
    :param feature_matrix:
    :param parameters: beta and alphas in one list
    :return: ingoing edge probabilities for each node
    """
    return sigmoid(feature_mat @ parameters)


def simulate_activations(G, seeds, edge_probs):
    activations = dict(zip(G.nodes() , [False]*len(G)))
    attempted_activations = np.zeros(len(G))

    new_activations = list(seeds)
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


def divide_rows_by_mean(feature_mat):
    for i in range(feature_mat.shape[0]):
        rowsum = np.sum(feature_mat[i, 1:])
        if rowsum > 0:
            feature_mat[i, 1:] = feature_mat[i, 1:]/rowsum


def get_training_data_from_trials(seeds, trials):
    training_data = []
    training_labels = []

    for j in range(trials):
        seeds = [1, 10, 20, 30, 40, 50, 60, 70]  # fixed seed set
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
    return training_data, training_labels


def append_training_data_from_trial(seeds, training_data, training_labels, feature_mat, current_edge_prob):

    activations, attempted_activations = simulate_activations(G, seeds, current_edge_prob)

    for node, attempts in enumerate(attempted_activations):
        if attempts > 0:
            training_data.append(feature_mat[node])
            training_labels.append(activations[node])

        if attempts > 1:
            for _ in range(int(attempts) - 1):
                training_data.append(feature_mat[node])
                training_labels.append(False)

    return training_data, training_labels


def greedy(G, feature_mat, messageSize, seeds, currentParameters):
    message = np.zeros(messageSize)
    message[0] = 1
    bestMessage = list(message)

    for i in range(messageSize):

        currentResult = 0
        index = 0
        for feature in range(1, feature_mat.shape(1)):
            index += 1
            if message[feature] != 1:
                message[feature] = 1
                if bestMessage[feature-1] != 1:
                    message[feature-1] = 0

            current_feature_mat = np.array(feature_mat) * message
            edge_probs = get_edge_probabilities(current_feature_mat, currentParameters)
            activations = simulate_activations(G, seeds, edge_probs)[0]
            result = sum(activations)

            if result > currentResult:
                currentResult = result
                bestIndex = index

        bestMessage[bestIndex] = 1

    return bestMessage





if __name__ == "__main__":
    numberOfFeatures = 200   ##3882
    numberOfNodes = 7420
    numberOfEdges = 115276
    numberOfTrials = 5
    seedset = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    logisticRegr = LogisticRegression(fit_intercept=False, warm_start=True)

    G, feature_matrix = prep.generate_graph_with_features(numberOfNodes, numberOfEdges, numberOfFeatures) #RandomGraph=False, filenameGraph = 'vk_mv.txt', filenameMember='vk_mem.txt')
    alphas = prep.generate_alphas(numberOfFeatures)
    beta = -4
    parameters = np.insert(alphas, 0, beta)

    feature_matrix = np.array(feature_matrix[:, :numberOfFeatures])  ##Extracting subset of features
    parameters = parameters[:numberOfFeatures]

    #DO AVERAGING ON ROWS SUCH THAT IT DOES NOT MATTER THAT YOU LIKE MANY FEATURES
    divide_rows_by_mean(feature_matrix)

    edge_probs = get_edge_probabilities(feature_matrix, parameters)
    print(edge_probs)
    print(np.array(edge_probs).mean(), np.array(edge_probs).max(), np.array(edge_probs).min())
    training_data = []
    training_labels = []
    averageProbabilityDetoriation = []
    current_params = np.ones(numberOfFeatures + 1)

    for _ in range(5):
        for k in range(numberOfTrials):
            msg = np.zeros(numberOfFeatures)
            msg[0] = 1
            msg[k + 1:(k + 1) * 40] = 1

            current_feature_matrix = np.array(feature_matrix) * msg
            current_probs = get_edge_probabilities(current_feature_matrix, parameters)

            print(np.array(current_probs).mean())

            labelLength = len(training_labels)
            training_data, training_labels = append_training_data_from_trial(seedset, training_data, training_labels,
                                                                             current_feature_matrix, current_probs)
            print(len(training_labels))

            logisticRegr.fit(training_data, training_labels)
            current_params = logisticRegr.coef_[0]

            predictedProbs = get_edge_probabilities(feature_matrix, logisticRegr.coef_[0])
            averageProbabilityDetoriation.append(np.mean(np.abs(edge_probs - predictedProbs)))

    print(averageProbabilityDetoriation)
    plt.plot(np.arange(numberOfTrials*2), averageProbabilityDetoriation, 'g^')
    plt.ylabel("Average error in probability estimate")
    plt.xlabel("Round")
    plt.savefig('example.pdf')
    plt.savefig('example.pgf')
    plt.show()



    """
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)

    print('shape of training data was: ', np.array(training_data).shape)

    print('max number of features: ', max(training_data.sum(axis=1)))

    print('average edge prob was: ', sum(edge_probs)/numberOfNodes)

    print('actual params: ', parameters)
    print('predicted params: ', logisticRegr.coef_)
    print('actual probs: ', edge_probs)
    predictedProbs = get_edge_probabilities(feature_matrix, logisticRegr.coef_[0])
    print('predicted probs: ', predictedProbs)
    print('On average parameter estimate is this far off: ', np.mean(np.abs(parameters-logisticRegr.coef_)))
    print('On average probability estimate was this far off: ', np.mean(np.abs(edge_probs-predictedProbs)))
    print('The maximal difference in probability estimate was', np.max(np.abs(edge_probs-predictedProbs)))
    #print(logisticRegr.intercept_)

    #print((logisticRegr.predict_proba(training_data)).mean())
    #print(mini_batch_grad_descent(training_data, training_labels))
    """