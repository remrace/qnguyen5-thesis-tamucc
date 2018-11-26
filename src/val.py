from data import *
import model
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import SegEval as ev
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy import stats

INPUT_SIZE = (481, 321, 3)
NUM_OUTPUTS = 1
KERNEL_SIZE = 5
N = (INPUT_SIZE[0]-1) * INPUT_SIZE[1] + (INPUT_SIZE[1]-1) * INPUT_SIZE[0]
D = KERNEL_SIZE * KERNEL_SIZE * 3 

def test():
    my_model = model.unet()
    my_model.load_weights('11-26.hdf5')
    data = TrainData()
    
    pred = my_model.predict([data[0], data[1]], batch_size=1)
    
    images, nlabels, elabels = data[0][0], data[1][0], data[2][0]
    
    G = nx.grid_2d_graph(elabels.shape[0], elabels.shape[1])
    for (u,v,d) in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1

        d['weight'] = elabels[u[0], u[1], channel]

    theta = 0.0
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}

    result_image = np.zeros((elabels.shape[0], elabels.shape[1]))

    for j in range(result_image.shape[1]):
        for i in range(result_image.shape[0]):
            result_image[i,j] = L[(i,j)]

    fig=plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 5, 1)
    plt.imshow(images)
    fig.add_subplot(1, 5, 2)
    plt.imshow(np.squeeze(nlabels), cmap='nipy_spectral')
    fig.add_subplot(1, 5, 3)
    plt.imshow(elabels[:,:,0], cmap='nipy_spectral')
    fig.add_subplot(1, 5, 4)
    plt.imshow(elabels[:,:,1], cmap='nipy_spectral')
    fig.add_subplot(1, 5, 5)
    plt.imshow(result_image, cmap='nipy_spectral')
    plt.show()
    
    

def test_aff(aff, nlabel, elabel, theta = 0.0, name=None):
    G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
    nlabels_dict = dict()
    correlation_clustering_error = 0.0
    
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1
        d['weight'] =  aff[0, u[0], u[1], channel]

        nlabels_dict[u] = nlabel[0, u[0], u[1], 0]
        nlabels_dict[v] = nlabel[0, v[0], v[1], 0]

        if d['weight'] * elabel[0, u[0], u[1], channel] < 0.0:
            correlation_clustering_error += abs(d['weight'])
    
    G.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    
    pred_label_dict = dict()
    for s, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            pred_label_dict[node] = s
    
    print('num of segs of ' + name, len(list(nx.connected_components(G))))
    print('rand error of ' + name, ev.FindRandErrorAtThreshold(WG=G, gtLabels=nlabels_dict, T=0.0))
    print('correlation_clustering_error of ' + name, correlation_clustering_error/(np.sum(np.abs(aff))))
    print('train loss of ' + name, tf.Session().run(loss(aff, nlabel, elabel)))
    result_image = np.zeros((480, 320))

    for node, label in pred_label_dict.items():
        result_image[node] = label
        #print(label)

    return result_image

def loss(y_pred, nlabels, elabels):
    def GetRandWeights(n, y):
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        for u, v, d in G.edges(data = True):
            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1
            d['weight'] =  y[0, u[0], u[1], channel]
            nlabels_dict[u] = n[0, u[0], u[1], 0]
            nlabels_dict[v] = n[0, v[0], v[1], 0]

        [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
        
        # for class imbalance
        posWeight = 0.0
        negWeight = 0.0
        # start off with every point in own cluster
        posError = totalPos
        negError = 0.0

        WY = np.zeros((1, 480, 320, 2), np.float32)
        SY = np.ones((1, 480, 320, 2), np.float32)
        for i in range(len(posCounts)):
            posError = posError - posCounts[i]
            negError = negError + negCounts[i]

            WS = posError - negError
        
            (u,v) = mstEdges[i]

            if u[0] == v[0]: #vertical, dy, channel 0
                channel = 0
            else: #horizontal, dx, channel 1
                channel = 1
            
            if WS > 0.0:
                WY[0, u[0], u[1], channel] = abs(WS) + posWeight
                if nlabels_dict[u] != nlabels_dict[v]:
                    SY[0, u[0], u[1], channel] = -1.0
            if WS < 0.0: 
                WY[0, u[0], u[1], channel] = abs(WS) + negWeight
                if nlabels_dict[u] == nlabels_dict[v]:
                    SY[0, u[0], u[1], channel] = -1.0
        
        # Std normalization
        totalW = np.sum(WY)
        WY = np.divide(WY, totalW)

        return WY, SY
    WY, SY = tf.py_func(GetRandWeights, [nlabels, y_pred], [tf.float32, tf.float32])
    newY = tf.multiply(SY, elabels)

    edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y_pred, newY)))

    weightedLoss = tf.multiply(WY, edgeLoss)

    return tf.reduce_sum(weightedLoss)

if __name__ == '__main__':
    print('Init')
    test()
    print('Exit')
