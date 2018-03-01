from __future__ import division
import networkx as nx
import numpy as np
import math
import graph_tool.all as gt
import time



def get_feature_matrix(filename, numberOfNodes, numberOfFeatures):
    """
    :param filename:
    :param output:
    :return: matrix with ones in first column and 0 or 1 in rest depending on if user likes feature or not
    """
    feature_matrix = np.zeros(numberOfFeatures*numberOfNodes).reshape(numberOfNodes, numberOfFeatures)

    featureToIndex = dict()
    featureCount = 0
    nodeCount = 0

    with open(filename) as f:
        for line in f:
            d = line.split()
            features = d[1:]
            for feature in features:
                if feature not in featureToIndex:
                    featureToIndex[feature] = featureCount
                    featureCount += 1

                feature_matrix[nodeCount, featureToIndex[feature]] +=1

            nodeCount +=1

    feature_matrix = np.insert(feature_matrix, 0, 1, axis=1)
    return feature_matrix

def generate_alphas(numberOfFeatures):
    alphas = np.random.randint(-5, 5, numberOfFeatures)
    #alphas = np.ones(numberOfFeatures)
    #for i in range(int(np.floor(numberOfFeatures/2))):
        #alphas[i*2] += 1
        #alphas[i*2+1] +=4
    return alphas

def read_graph(filename, directed=True):
    """
    Create networkx graph reading file.
    :param filename: every line (u, v)
    :param directed: boolean
    :return:
    """
    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            d = line.split()
            G.add_edge(int(d[0]), int(d[1]))
    return G


if __name__ == "__main__":
    print(get_feature_matrix('vk_mem.txt', 7420, 3882))
    print(generate_alphas(3882))
    print(sum(get_feature_matrix('vk_mem.txt', 7420, 3882)))