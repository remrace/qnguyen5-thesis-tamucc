import tensorflow as tf
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import networkx as nx
import SegEval as ev

import tensorflow.contrib as tfcontrib
from keras.optimizers import *
from keras import layers
from keras import losses
from keras import models
from keras import backend as K
from scipy import signal
INPUT_SIZE = (480, 320, 3)



def rand_weight(y_pred, nlabels):
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
        SY = np.zeros((1, 480, 320, 2), np.float32)
        for i in range(len(posCounts)):
            #posError = posError - posCounts[i]
            #negError = negError + negCounts[i]

            #WS = posError - negError

            WS = posCounts[i] - negCounts[i]

            (u,v) = mstEdges[i]

            if u[0] == v[0]: #vertical, dy, channel 0
                channel = 0
            else: #horizontal, dx, channel 1
                channel = 1
            
            if WS > 0.0:
                WY[0, u[0], u[1], channel] = abs(WS) + posWeight
                #if nlabels_dict[u] != nlabels_dict[v]:
                    #SY[0, u[0], u[1], channel] = -1.0
                SY[0, u[0], u[1], channel] = 1.0
            if WS < 0.0: 
                WY[0, u[0], u[1], channel] = abs(WS) + negWeight
                #if nlabels_dict[u] == nlabels_dict[v]:
                    #SY[0, u[0], u[1], channel] = -1.0
                SY[0, u[0], u[1], channel] = -1.0
        
        # Std normalization


        return WY, SY
    WY, SY = tf.py_func(GetRandWeights, [nlabels, y_pred], [tf.float32, tf.float32])
    return [WY, SY]
'''

def rand_weight(y_pred, nlabels):
    def GetRandWeights(y, n):
        WY = np.zeros( (1, N, 1), np.float32)
        SY = np.ones( (1, N, 1), np.float32)
        
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        edgeInd = dict()
        upto = 0
        for u, v, d in G.edges(data = True):
            d['weight'] = y[0, upto, 0]
            edgeInd[(u,v)] = upto
            nlabels_dict[u] = n[0, u[0], u[1], 0]
            nlabels_dict[v] = n[0, v[0], v[1], 0]
            upto = upto + 1

        [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
        
        # for class imbalance
        posWeight = 0.0
        negWeight = 0.0
        # start off with every point in own cluster
        posError = totalPos
        negError = 0.0

        
        for i in range(len(posCounts)):
            posError = posError - posCounts[i]
            negError = negError + negCounts[i]

            WS = posError - negError
            
            ind = edgeInd[ mstEdges[i] ]
            WY[0, ind] = np.abs(WS)

            
            if WS > 0.0:
                if nlabels_dict[u] != nlabels_dict[v]:
                    SY[0, ind, 0] = -1.0
                
            if WS < 0.0: 
                if nlabels_dict[u] == nlabels_dict[v]:
                    SY[0, ind, 0] = -1.0
        
        # Std normalization
        totalW = np.sum(WY)
        WY = WY / totalW

        return WY, SY
    WY, SY = tf.py_func(GetRandWeights, [y_pred, nlabels], [tf.float32, tf.float32])

    return [WY, SY]

'''
def output_shape(input_shape):
	return [input_shape, input_shape]

def ff_loss(WY, SY):
    SY = tf.reshape(SY, [-1])
    WY = tf.reshape(WY, [-1])
    WY = tf.divide(WY, tf.reduce_sum(WY))
    def f_loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        #newY = tf.maximum(SY, y_true)

        edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y_pred, SY)))

        weightedLoss = tf.multiply(WY, edgeLoss)

        return tf.reduce_sum(weightedLoss)
    return f_loss



def unet(pretrained_weights=None):
    input_image = layers.Input(shape=(INPUT_SIZE), name='input_image')
    input_nlabels = layers.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1],1), name='input_nlabels')
    
    '''
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(input_image)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = layers.Concatenate(axis = 3)([drop4,up6])
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = layers.Concatenate(axis = 3)([conv3,up7])
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = layers.Concatenate(axis = 3)([conv2,up8])
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = layers.Concatenate(axis = 3)([conv1, up9])
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = layers.Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    '''
    #transform outputs to shape (?, 204160, 1)
   
    aff00 = layers.Conv2D(filters = 1, kernel_size = 25, padding = 'same', data_format='channels_last', activation = 'relu')(input_image)
    aff10 = layers.Conv2D(filters = 1, kernel_size = 25, padding = 'same', data_format='channels_last', activation = 'relu')(input_image)
    aff01 = layers.Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(aff00)
    aff11 = layers.Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(aff10)
    aff02 = layers.Conv2D(1, 1, padding = 'same')(aff01)
    aff12 = layers.Conv2D(1, 1, padding = 'same')(aff11)
    aff = layers.Concatenate(axis = -1)([aff02,aff12])

    WY, SY = layers.Lambda(rand_weight, output_shape=output_shape, arguments={'nlabels': input_nlabels})(aff)



    model = models.Model(inputs=[input_image, input_nlabels], outputs = [aff])
    model.summary()
    model.compile(optimizer = SGD(lr=0.1), loss = ff_loss(WY, SY))
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    print("Init")
    model = unet()
    print("Exit")