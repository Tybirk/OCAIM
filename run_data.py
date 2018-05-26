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
from timeit import default_timer as timer


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


def get_edge_probabilities(feature_mat, parameters, linear = False, hybrid=False, linear_parameters = None):
    """
    :param feature_matrix:
    :param parameters: beta and alphas in one list
    :return: ingoing edge probabilities for each node
    """
    if hybrid:
        mixed_probs = 0.45 * sigmoid(feature_mat @ parameters) + 0.55 * np.array([feature_mat[i, :]@linear_parameters for i in range(feature_mat.shape[0])])
        print(0.45*np.sum(sigmoid(feature_mat @ parameters)))
        print(0.55*np.sum(np.array([feature_mat[i, :]@linear_parameters for i in range(feature_mat.shape[0])])))
        return np.maximum(np.minimum(mixed_probs, 1), 0)
    if not linear:
        return sigmoid(feature_mat @ parameters)
    else:
        edge_probs = [max(min(feature_mat[i, :]@linear_parameters, 1),0) for i in range(feature_mat.shape[0])]
        return edge_probs


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


def append_training_data_from_trial(G, seeds, training_dat, training_labs, feature_mat, current_edge_prob):

    activations, attempted_activations = simulate_activations(G, seeds, current_edge_prob)

    for node, attempts in enumerate(attempted_activations):
        if attempts > 0:
            training_dat.append(feature_mat[node])
            training_labs.append(activations[node])

        if attempts > 1:
            for _ in range(int(attempts) - 1):
                training_dat.append(feature_mat[node])
                training_labs.append(False)

    return training_dat, training_labs


def get_parameter_std_deviation(training_dat, pred_probs):
    ##Return standard deviation if possible, otherwise set standard deviation large
    #training_dat.sum(axis=1)

    training_dat = np.array(training_dat)
    pred_probas = np.array(pred_probs)
    V = pred_probas*(1-pred_probas)
    print(training_dat.shape)
    print(V.shape)
    cov_mat = np.linalg.inv(training_dat.T @ (training_dat * V[:,np.newaxis]))
    print('Standard devs: ', np.sqrt(np.diag(cov_mat)).sum())

    return np.sqrt(np.diag(cov_mat))

    #https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients


def get_initial_training_data(G, featuresPerTrial, feature_mat, true_parameters, logistic_reg, training_dat = [], training_labs = []):
    numberOfFeatures = np.array(feature_mat).shape[1]

    numberOfRounds = int(math.ceil(numberOfFeatures/featuresPerTrial))
    for k in range(numberOfRounds):
        msg = np.zeros(numberOfFeatures)
        msg[0] = 1
        if (k+1) * featuresPerTrial + 1 > numberOfFeatures:
            msg[k*featuresPerTrial + 1: numberOfFeatures] = 1
            msg[1: (k+1) * featuresPerTrial + 1 - numberOfFeatures] = 1
        else:
            msg[k*featuresPerTrial + 1: ((k + 1) * featuresPerTrial) + 1] = 1

        current_feature_matrix = np.array(feature_mat) * msg
        current_probs = get_edge_probabilities(current_feature_matrix, true_parameters)

        training_dat, training_labs = append_training_data_from_trial(G, seedset, training_dat, training_labs,
                                                                             current_feature_matrix, current_probs)

    return training_dat, training_labs


def exploitation_only(G, feature_mat, message_size, seeds, numberOfTrials, true_parameters, logistic_reg, training_dat, training_labs, numberOfMCsims = 1):
    sizeOfInitialData = sum(training_labs)
    spreadData = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        logistic_reg.fit(training_dat, training_labs)
        current_params = logisticRegr.coef_[0]
        message = greedy(G,feature_mat, message_size, seeds, current_params, numberOfMCsims)

        print('message was: ', np.where(np.array(message) > 0))

        current_feature_mat = np.array(feature_mat) * message
        current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters)

        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs, current_feature_mat, current_edge_prob)
        print('Total spread after {0} trials was: '.format(i), sum(training_labs) - sizeOfInitialData)
        spreadData[i] = sum(training_labs) - sizeOfInitialData
    return current_params, training_dat, training_labs, spreadData


def explore_and_exploit(G, feature_mat, message_size, seeds, numberOfTrials, true_parameters, logistic_reg, training_dat, training_labs, numberOfMCsims = 1, c=1, hybrid=False):
    sizeOfInitialData = sum(training_labs)
    spreadData = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        logistic_reg.fit(training_dat, training_labs)
        probs = logistic_reg.predict_proba(training_dat)
        probs = probs[:, 0]

        standard_dev = get_parameter_std_deviation(training_dat, probs)
        current_params = logisticRegr.coef_[0] + c*standard_dev

        message = greedy(G,feature_mat, message_size, seeds, current_params, numberOfMCsims, hybrid)

        print('message was: ', np.where(np.array(message) > 0))

        current_feature_mat = np.array(feature_mat) * message
        current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters)

        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs, current_feature_mat, current_edge_prob)
        print('Total spread after {0} trials was: '.format(i), sum(training_labs) - sizeOfInitialData)
        spreadData[i] = sum(training_labs) - sizeOfInitialData
    return current_params, training_dat, training_labs, spreadData

def exploit_perfect_knowledge(G, feature_mat, message_size, seeds, numberOfTrials, true_parameters, numberOfMCsims = 1, c = 1):
    spreadData = np.zeros(numberOfTrials)
    training_dat = []
    training_labs = []
    message = greedy(G, feature_mat, message_size, seeds, true_parameters, numberMCsims=100)
    print('best message was: ', np.where(np.array(message) > 0))
    current_feature_mat = np.array(feature_mat) * message
    current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters)
    for i in range(numberOfTrials):

        #message = greedy(G, feature_mat, message_size, seeds, true_parameters, numberOfMCsims)
        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs,
                                                                      current_feature_mat, current_edge_prob)
        print('Total spread after {0} trials was: '.format(i), sum(training_labs))
        spreadData[i] = sum(training_labs)
    return training_dat, training_labs, spreadData


def greedy(G, feature_mat, messageSize, seeds, currentParameters, numberMCsims = 1, linear=False, UCB=False):
    numberFeatures = feature_mat.shape[1]
    message = np.zeros(numberFeatures)
    message[0] = 1 ##bias term
    bestMessage = list(message)

    for i in range(messageSize):
        currentResult = 0
        index = 0
        message = list(bestMessage)
        for feature in range(1, numberFeatures):
            index += 1

            if bestMessage[feature-1] != 1:
                message[feature-1] = 0

            if message[feature] != 1:
                message[feature] = 1

                activations = []

                assert(sum(message) == 2 + i)
                current_feature_mat = np.array(feature_mat) * message
                if not linear:
                    if UCB:
                        theta, standard_devs, sigma, c = currentParameters
                        edge_prob = [sigmoid(current_feature_mat[i, :].T @ theta + c * np.sqrt(current_feature_mat[i, :].T @ standard_devs @ current_feature_mat[i, :])) for i in range(feature_mat.shape[0])]
                    else:
                        edge_prob = get_edge_probabilities(current_feature_mat, currentParameters)
                else:
                    theta, M_inverse, sigma, c = currentParameters
                    edge_prob = [max(min((current_feature_mat[i, :].T @ theta + c * np.sqrt(current_feature_mat[i, :].T @ M_inverse @ current_feature_mat[i, :])), 1), 0) for i in range(feature_mat.shape[0])]
                for _ in range(numberMCsims):
                    activations.append(simulate_activations(G, seeds, edge_prob)[0].values())
                result = sum(activations[0])/numberMCsims

                if result > currentResult:
                    currentResult = result
                    bestIndex = index

        bestMessage[bestIndex] = 1
    #print(edge_prob)
    print(np.max(edge_prob))
    return bestMessage


def OCAIMLinUCB(G, feature_mat, message_size, seeds, number_of_trials, true_parameters, training_dat, training_labs, number_mc_sims=1, sigma=4, c=1, hybrid=False, linear_parameters = None):
    sizeOfInitialData = sum(training_labs)
    spreadData = np.zeros(numberOfTrials)
    n = feature_mat.shape[1]
    B = np.zeros(n)
    M = np.identity(n)
    M_inverse = np.identity(n)
    prob_detoriation = np.zeros(numberOfTrials)
    ##Find initial M and B
    for i, edge in enumerate(np.array(training_dat)):
        M_inverse -= (M_inverse @ np.outer(edge, edge) @ M_inverse) / (edge.T @ M_inverse @ edge + sigma ** 2)
        # M += 1/sigma**2 * edge @ edge.T
        if training_labs[i]:
            B += edge

    for j in range(number_of_trials):
        #M_inverse = M_inverse - (M_inverse*)
        theta = 1/sigma**2 * M_inverse @ B
        current_params = [theta, M_inverse, sigma, c]
        message = greedy(G, feature_mat, message_size, seeds, current_params, numberMCsims= number_mc_sims, linear=True)

        print('message was: ', np.where(np.array(message) > 0))

        current_feature_mat = np.array(feature_mat) * message
        current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters, linear=True, hybrid=hybrid, linear_parameters=linear_parameters)
        #prob_detoriation[j] = np.mean(np.abs(get_edge_probabilities(current_feature_mat, current_params[0], linear=True) - current_edge_prob))

        len_old_training_data = len(training_labs)

        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs,
                                                                      current_feature_mat, current_edge_prob)
        print('Total spread after {0} trials was: '.format(j), sum(training_labs) - sizeOfInitialData)
        spreadData[j] = sum(training_labs) - sizeOfInitialData

        for i, edge in enumerate(np.array(training_dat)[len_old_training_data:, :]):
            M_inverse -= (M_inverse @ np.outer(edge, edge) @ M_inverse)/(edge.T @ M_inverse @ edge + sigma**2)
            #M += 1/sigma**2 * edge @ edge.T
            if training_labs[len_old_training_data+i]:
                B += edge

    return training_dat, training_labs, spreadData #, prob_detoriation

def explore_and_exploit_2(G, feature_mat, message_size, seeds, numberOfTrials, true_parameters, logistic_reg, training_dat, training_labs, numberOfMCsims = 1, c=1, sigma=1, hybrid=False, linear_parameters = None):
    sizeOfInitialData = sum(training_labs)
    n = feature_mat.shape[1]
    prob_detoriation = np.zeros(numberOfTrials)
    spreadData = np.zeros(numberOfTrials)
    standard_dev = np.identity(n)
    if len(training_labs) > 0:
        for edge in np.array(training_dat):
            standard_dev -= (standard_dev @ np.outer(edge, edge) @ standard_dev)/(edge.T @ standard_dev @ edge + sigma**2) 
            #np.linalg.inv(sigma**(-2)*sum([np.outer(edge, edge) for edge in training_dat]))
   
    for j in range(numberOfTrials):
        if j > 0 or len(training_labs) > 0:
            logistic_reg.fit(training_dat, training_labs)
            current_params = [logisticRegr.coef_[0], standard_dev, sigma, c]
        else:
            current_params = [np.zeros(n), standard_dev, sigma, c]

        message = greedy(G,feature_mat, message_size, seeds, current_params, numberOfMCsims, linear=False, UCB=True)

        print('message was: ', np.where(np.array(message) > 0))

        current_feature_mat = np.array(feature_mat) * message
        current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters, hybrid=hybrid, linear_parameters=linear_parameters)
        prob_detoriation[j] = np.mean(np.abs(get_edge_probabilities(current_feature_mat, current_params[0]) - current_edge_prob))
        len_old_training_data = len(training_labs)
        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs, current_feature_mat, current_edge_prob)

        #for i, edge in enumerate(np.array(training_dat)[len_old_training_data:, :]):
        for edge in np.array(training_dat)[len_old_training_data:, :]:
            standard_dev -= (standard_dev @ np.outer(edge, edge) @ standard_dev)/(edge.T @ standard_dev @ edge + sigma**2)
        #M += 1/sigma**2 * edge @ edge.T
       

        print('Total spread after {0} trials was: '.format(j), sum(training_labs) - sizeOfInitialData)
        spreadData[j] = sum(training_labs) - sizeOfInitialData
    return training_dat, training_labs, spreadData, prob_detoriation

def random_messages(G, feature_mat, message_size, seeds, numberOfTrials, true_parameters):
    spreadData = np.zeros(numberOfTrials)
    training_labs = []
    training_dat = []
    for j in range(numberOfTrials):
        message = np.zeros(feature_mat.shape[1])
        message_idx = random.sample(range(1, feature_mat.shape[1]), message_size)
        message[message_idx] = 1
        message[0] = 1
        print('message was: ', np.where(np.array(message) > 0))
        current_feature_mat = np.array(feature_mat) * message
        current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters)
        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs, current_feature_mat, current_edge_prob)
        spreadData[j] = sum(training_labs)
        print('Total spread after {0} trials was: '.format(j), sum(training_labs))
    return spreadData

def exploitation_only_pure(G, feature_mat, message_size, seeds, numberOfTrials, true_parameters, logistic_reg, training_dat, training_labs, numberOfMCsims = 1):
    sizeOfInitialData = sum(training_labs)
    spreadData = np.zeros(numberOfTrials)
    logistic_reg.fit(training_dat, training_labs)
    current_params = logisticRegr.coef_[0]
    message = greedy(G,feature_mat, message_size, seeds, current_params, 100)
    current_feature_mat = np.array(feature_mat) * message
    current_edge_prob = get_edge_probabilities(current_feature_mat, true_parameters)
    for i in range(numberOfTrials):
        print('message was: ', np.where(np.array(message) > 0))
        training_dat, training_labs = append_training_data_from_trial(G, seeds, training_dat, training_labs, current_feature_mat, current_edge_prob)
        print('Total spread after {0} trials was: '.format(i), sum(training_labs) - sizeOfInitialData)
        spreadData[i] = sum(training_labs) - sizeOfInitialData
    return current_params, training_dat, training_labs, spreadData

if __name__ == "__main__":
    number_of_possible_features = 3882
    numberOfFeatures = 50 ##3882
    numberOfNodes = 7420
    numberOfEdges = 115276
    numberOfTrials = 100
    number_of_MC_sims = 10
    messageSize = 2
    seedset = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    logisticRegr = LogisticRegression(fit_intercept=False, warm_start=True)

    #G, feature_matrix = prep.generate_graph_with_features(numberOfNodes, numberOfEdges, numberOfFeatures) #RandomGraph=False, filenameGraph = 'vk_mv.txt', filenameMember='vk_mem.txt')
    G, feature_matrix = prep.generate_graph_with_features(numberOfNodes, numberOfEdges,
                                                        number_of_possible_features, RandomGraph=False,
                                                        filenameGraph = 'vk_mv.txt', filenameMember='vk_mem.txt')
    random.seed(10)
    alphas = prep.generate_alphas(numberOfFeatures)
    beta = -4  #-4 approx equal to 0.018    -3: ##Approx equal 0.0474
    parameters = np.insert(alphas, 0, beta)
    true_parameters = list(parameters)
    beta_lin = 0.018 #0.0474
    parameters_lin = np.insert(alphas/10, 0, beta_lin)

    #feature_matrix = np.array(feature_matrix[:, :(numberOfFeatures+1)])  ##Extracting subset of features
    feature_counts = np.sum(feature_matrix, axis=0)
    idx_most_common = feature_counts.argsort()[-(numberOfFeatures+1):][::-1]
    feature_matrix = feature_matrix[:, idx_most_common]
    # print(feature_matrix)
    
    #constant = np.sum(np.array([feature_matrix[i, :]@parameters_lin for i in range(feature_matrix.shape[0])]))/np.sum(sigmoid(feature_matrix @ parameters))
    #print(parameters_lin)

    #DO AVERAGING ON ROWS SUCH THAT IT DOES NOT MATTER THAT YOU LIKE MANY FEATURES
    #divide_rows_by_mean(feature_matrix)
    #print(get_edge_probabilities(feature_matrix, parameters, hybrid=True, linear_parameters=parameters_lin))
    #print("Avg edge prob including all features was: ", np.mean(get_edge_probabilities(feature_matrix, parameters, hybrid=True, linear_parameters=parameters_lin)))
    """
    start = timer()
    training_data, training_labels, spreadData1, prob_detoriation1 = explore_and_exploit_2(G, feature_matrix, messageSize, seedset, numberOfTrials,
                                                              parameters, logisticRegr, [], [], hybrid=True, linear_parameters=parameters_lin, numberOfMCsims=number_of_MC_sims)
    end = timer()
    time1 = end - start
    start = timer()
    training_data, training_labels, spreadData2, prob_detoriation2 = OCAIMLinUCB(G, feature_matrix, messageSize, seedset, numberOfTrials, parameters, [], [], hybrid=True, linear_parameters=parameters_lin, number_mc_sims=number_of_MC_sims)
    end = timer()
    time2 = end - start
    print(time1, time2)
    print(spreadData1)
    print(spreadData2)
    print(prob_detoriation1)
    print(prob_detoriation2)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(numberOfTrials), spreadData1, 'g^')
    plt.plot(np.arange(numberOfTrials), spreadData2, 'bs')
    plt.subplot(212)
    plt.plot(np.arange(numberOfTrials), prob_detoriation1, 'g^')
    plt.plot(np.arange(numberOfTrials), prob_detoriation2, 'bs')
    plt.show() """

    """not_invertible = True
    while not_invertible:
        try:

            matrix = np.linalg.inv(sum([np.outer(edge, edge) for edge in training_data]))
            not_invertible = False
        except:
            pass"""
            
    for i in range(3):
        random.seed(1)
        messageSize = i+1
        training_data, training_labels = get_initial_training_data(G, messageSize, feature_matrix, parameters, logisticRegr)
        training_data_2 = list(training_data)
        training_labels_2 = list(training_labels)
        training_data_3 = list(training_data)
        training_labels_3 = list(training_labels)
        training_data_4 = list(training_data)
        training_labels_4 = list(training_labels)
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=True)  ##RESET
        
        spreadData6 = exploitation_only_pure(G, feature_matrix, messageSize, seedset, numberOfTrials, parameters,
                                                                                        logisticRegr, training_data_4,
                                                                                        training_labels_4,
                                                                                        numberOfMCsims=number_of_MC_sims)[3]

        #current_params, training_data_3, training_labels_3, 
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=False)
        logisticRegr.fit(training_data_3, training_labels_3)
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=True)
        spreadData4 = explore_and_exploit(G, feature_matrix,
                                                                                                messageSize, seedset,
                                                                                                numberOfTrials, parameters,
                                                                                                logisticRegr,
                                                                                                training_data_3,
                                                                                                training_labels_3,
                                                                                                numberOfMCsims=number_of_MC_sims,
                                                                                                c=1)[3]
        
    
        
        spreadData5 = random_messages(G, feature_matrix, messageSize, seedset, numberOfTrials, parameters)
        
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=False)
        logisticRegr.fit(training_data_2, training_labels_2)
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=True)
        #training_data_2, training_labels_2, 
        spreadData2 = explore_and_exploit_2(G, feature_matrix,
                                                                                                messageSize, seedset,
                                                                                                numberOfTrials, parameters,
                                                                                                logisticRegr,
                                                                                                training_data_2,
                                                                                                training_labels_2,
                                                                                                numberOfMCsims=number_of_MC_sims,
                                                                                                c=1)[2]
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=False)
        logisticRegr.fit(training_data, training_labels)
        logisticRegr = LogisticRegression(fit_intercept=False, warm_start=True)
        
        #current_params, training_data, training_labels, 
        spreadData1 = exploitation_only(G, feature_matrix, messageSize, seedset, numberOfTrials, parameters,
                                                                                        logisticRegr, training_data,
                                                                                        training_labels,
                                                                                        numberOfMCsims=number_of_MC_sims)[3]
    
        
        #training_data_4, training_labels_4, 
        spreadData3 = exploit_perfect_knowledge(G, feature_matrix,
                                                                                messageSize, seedset,
                                                                                numberOfTrials, true_parameters,
                                                                                number_of_MC_sims)[2]
        

    
    
    
        #training_data_3, training_labels_3, spreadData4 = OCAIMLinUCB(G, feature_matrix, messageSize,
         #                                                                            seedset, numberOfTrials, parameters,
          #                                                                           training_data_3, training_labels_3, hybrid=False,
           #                                                                          linear_parameters=parameters_lin,
            #                                                                         number_mc_sims=number_of_MC_sims)
    
        plt.plot(np.arange(numberOfTrials), spreadData1, 'b:') #g^:  exploit
        plt.plot(np.arange(numberOfTrials), spreadData2, 'g--') #bs-- expl_exploit2
        plt.plot(np.arange(numberOfTrials), spreadData3, 'r-') #rD-. perfect
        plt.plot(np.arange(numberOfTrials), spreadData4, 'c-.') #cp-- expl:exploit1
        plt.plot(np.arange(numberOfTrials), spreadData5, ':') #random
        plt.plot(np.arange(numberOfTrials), spreadData6, 'y-.') #badexploit
        plt.ylabel("Accumulated Spread")
        plt.xlabel("Round")
        plt.savefig('example_{}.pdf'.format(i+1))
        #plt.savefig('example_{}.pgf'.format(i+1))
        plt.show()
        
        
        
        
        

    """
    edge_probs = get_edge_probabilities(feature_matrix, parameters)
    print(edge_probs)
    print(np.array(edge_probs).mean(), np.array(edge_probs).max(), np.array(edge_probs).min())

    averageProbabilityDetoriation = []
    current_params = np.ones(numberOfFeatures + 1)
    """
    """
    for _ in range(1):
        for k in range(numberOfTrials):
            msg = np.zeros(numberOfFeatures)
            msg[0] = 1
            msg[k + 1:(k + 1) * 10] = 1


            current_feature_matrix = np.array(feature_matrix) * msg
            current_probs = get_edge_probabilities(current_feature_matrix, parameters)

            print('The average edge probabilities with the current message: ', np.array(current_probs).mean())

            labelLength = len(training_labels)
            print('Total number of activations after {0} rounds was: '.format(k), sum(training_labels))
            training_data, training_labels = append_training_data_from_trial(seedset, training_data, training_labels,
                                                                             current_feature_matrix, current_probs)

            logisticRegr.fit(training_data, training_labels)
            current_params = logisticRegr.coef_[0]

            averageProbabilityDetoriation.append(np.mean(np.abs(parameters - current_params)))


    probs = logisticRegr.predict_proba(training_data)
    probs = probs[:,0]

    standard_dev = get_parameter_std_deviation(np.array(training_data), np.array(probs))
    print('standard dev: ', standard_dev)


    bestMsg = greedy(G, feature_matrix, 2, seedset, current_params, number_of_MC_sims)

    print('length of best msg: ', sum(bestMsg))

    print(bestMsg)
    current_feature_matrix = np.array(feature_matrix) * bestMsg

    bestMsgEdgeProbs = get_edge_probabilities(current_feature_matrix, parameters)
    print(np.array(bestMsgEdgeProbs).mean())
    activated = simulate_activations(G, seedset, bestMsgEdgeProbs)[0].values()
    print('Totally activated: ', sum(activated))

    print(averageProbabilityDetoriation)
    plt.plot(np.arange(numberOfTrials*1), averageProbabilityDetoriation, 'g^')
    plt.ylabel("Average error in probability estimate")
    plt.xlabel("Round")
    plt.savefig('example.pdf')
    plt.savefig('example.pgf')
    plt.show()


    """
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