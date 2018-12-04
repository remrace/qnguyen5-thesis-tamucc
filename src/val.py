from data import *
import model
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import SegEval as ev
import SegGraph as seglib
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy import stats
INPUT_SIZE = (480, 320, 3)
NUM_EDGES = INPUT_SIZE[0]*(INPUT_SIZE[1]-1) + (INPUT_SIZE[0]-1)*INPUT_SIZE[1]
def test():
    my_model = model.unet(pretrained_weights='C:/model/12-2-2018.hdf5')
    data = ValData(1)
    
    pred = my_model.predict([data[0], data[1]], batch_size=1)
    
    images, nlabels, elabels = data[0][0], data[1][0], data[2][0]
    print('Got data')
    G = nx.grid_2d_graph(elabels.shape[0], elabels.shape[1])
    for (u,v,d) in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1

        d['weight'] = pred[0, u[0], u[1], channel]
    print('Got graph')
    W = sorted([w for (u,v,w) in G.edges(data = 'weight')])
    lowIndex = int(0.25*len(W))
    highIndex = int(0.75*len(W))
    #theta = [W[lowIndex], 0.0, W[highIndex]]
    theta = [-1]
    print(theta)
    minE = 0.0
    minT = 0.0
    for t in theta:
        L = seglib.GetLabelsAtThreshold(G,t)
        E = seglib.GetLabelEnergy(G,L)
        if E < minE:
            minT = t
            minE = E
            minL = L
    print('bestT: ', minT, 'minE: ', minE)

    L = minL

    result_image = np.zeros((elabels.shape[0], elabels.shape[1]), np.float32)

    for j in range(result_image.shape[1]):
        for i in range(result_image.shape[0]):
            result_image[i,j] = L[(i,j)]

    fig=plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 5, 1)
    plt.imshow(np.squeeze(images))
    fig.add_subplot(1, 5, 2)
    plt.imshow(np.squeeze(nlabels), cmap='nipy_spectral')
    fig.add_subplot(1, 5, 3)
    plt.imshow(pred[0, :,:,0], cmap='nipy_spectral')
    fig.add_subplot(1, 5, 4)
    plt.imshow(pred[0, :,:,1], cmap='nipy_spectral')
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
