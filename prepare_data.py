from __future__ import division
import networkx as nx
import numpy as np
import random
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

    featureToIndex = {}
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
                if featureToIndex[feature] < numberOfFeatures:
                    feature_matrix[nodeCount, featureToIndex[feature]] +=1

            nodeCount +=1

    feature_matrix = np.insert(feature_matrix, 0, 1, axis=1)
    return feature_matrix


def generate_alphas(numberOfFeatures):
    #alphas = np.random.randint(-4, 3, numberOfFeatures)
    #alphas = np.random.uniform(-5, 5, numberOfFeatures)
    alphas = np.ones(numberOfFeatures)
    alphas[0:5] = -2
    alphas[5:10] = -1.5
    alphas[10:15] = -1
    alphas[15:25] = 1
    alphas[25:30] = 2
    alphas[30:35] = 3
    alphas[35-40] = 1.5
    alphas[40-44] = 1.5
    alphas[44-47] = 2.5
    alphas[47-50] = -2.5
    
    #alphas[0:5] = -3
    #alphas[5:10] = -2
    #alphas[10:15] = -1
    #alphas[15:25] = 1
    #alphas[25:35] = 2
    #alphas[35:40] = 3
    #alphas[40-44] = 4
    #alphas[44-47] = 5
    #alphas[47-50] = -4
    #alphas = np.random.uniform(-1, 1, numberOfFeatures)
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
    G = nx.DiGraph() if directed else nx.Graph()
    with open(filename) as f:
        for line in f:
            d = line.split()
            G.add_edge(int(d[0]), int(d[1]))
    return G


def create_graph(numberOfNodes, numberOfEdges):
    G = nx.DiGraph()
    for _ in range(numberOfEdges):
        rand1 = random.randint(0, numberOfNodes-1)
        rand2 = random.randint(0, numberOfNodes-1)
        if rand1 != rand2:
            G.add_edge(rand1, rand2)
    return G


def create_feature_matrix(numberOfNodes, numberOfFeatures):
    #feature_matrix = [random.randint(0,1) for i in range(numberOfNodes) for j in range(numberOfFeatures)]

    feature_matrix = np.zeros(numberOfFeatures * numberOfNodes).reshape(numberOfNodes, numberOfFeatures)
    for i in range(numberOfNodes):
        for j in range(numberOfFeatures):
            if random.random() < 0.2:
                feature_matrix[i,j] = 1

    feature_matrix = np.insert(feature_matrix, 0, 1, axis=1) #First column consists on ones
    return feature_matrix


def generate_graph_with_features(numberOfNodes, numberOfEdges, numberOfFeatures, RandomGraph = True, filenameGraph=None, filenameMember = None):
    if RandomGraph:
        G = create_graph(numberOfNodes, numberOfEdges)
        feature_matrix = create_feature_matrix(numberOfNodes, numberOfFeatures)
    else:
        G = read_graph(filenameGraph)
        feature_matrix = get_feature_matrix(filenameMember, numberOfNodes, numberOfFeatures)
    return G, feature_matrix

if __name__ == "__main__":
    print(get_feature_matrix('vk_mem.txt', 7420, 3882))
    print(generate_alphas(3882))
    print(sum(get_feature_matrix('vk_mem.txt', 7420, 3882))[1:])
    G = read_graph('vk_mv.txt', directed=True)
    Gc = max(nx.strongly_connected_component_subgraphs(G), key=len)
    print(len(Gc))
    print([len(Gc) for Gc in sorted(nx.strongly_connected_component_subgraphs(G), key=len, reverse=True)])
    H = G.subgraph(np.arange(2000))
    print(len(H))
    print([len(Hc) for Hc in sorted(nx.strongly_connected_component_subgraphs(H), key=len, reverse=True)])
    #print(H.degree())
    