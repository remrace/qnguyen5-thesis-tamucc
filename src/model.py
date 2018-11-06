import tensorflow as tf
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import networkx as nx
import SegEval as ev
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
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
        if totalW > 0.0:
            WY = WY / totalW

        return WY, SY
    WY, SY = tf.py_func(GetRandWeights, [nlabels, y_pred], [tf.float32, tf.float32])
    return [WY, SY]

def output_shape(input_shape):
	return [input_shape, input_shape]

def ff_loss(WY, SY):
    def f_loss(y_true, y_pred):
        newY = tf.multiply(SY, y_true)

        edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y_pred, newY)))

        weightedLoss = tf.multiply(WY, edgeLoss)

        return tf.reduce_sum(weightedLoss)
    return f_loss


def unet(pretrained_weights=None):
    input_image = Input(shape=(480, 320, 3), name='input_image')
    input_nlabels = Input(shape=(480, 320, 1), name='input_nlabels')
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_image)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis = 3)([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis = 3)([conv3,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis = 3)([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis = 3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(2, 1, activation = 'tanh')(conv9)
    
    WY, SY = Lambda(rand_weight, output_shape=output_shape, arguments={'nlabels': input_nlabels})(conv10)


    model = Model(input=[input_image, input_nlabels], output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = ff_loss(WY, SY))

    

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    print("Init")
    model = unet()
    print("Exit")