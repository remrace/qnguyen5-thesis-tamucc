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

def rand_loss_function(y_true, y_pred):
    #y_true is the ground truth labels at pixel level
    #y_pred is the output of the network, currently having the same shape of the original image
    #calculating affinity edges ground truth edges(2 channels)
    #affinity edges 
    kernels = [[[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]]]
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = tf.constant(kernels, dtype=y_pred.dtype)

    gt_edges = tf.clip_by_value(tf.abs(tf.nn.depthwise_conv2d(input=y_true, filter=kernels_tf, strides=[1, 1, 1, 1], padding='SAME')), clip_value_min=0.0, clip_value_max=1.0)
    gt_edges = tf.subtract(1.0, tf.multiply(2.0, gt_edges))
    #print(affs.shape, gt_edges.shape)

    def GetRandWeights(y_true, y_pred):
        #print(y_true.shape, y_pred.shape)
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels = dict()
        for u, v, d in G.edges(data = True):
            d['weight'] =  np.subtract(1.0, np.multiply(2.0, np.abs(y_pred[0,u[0],u[1],0] - y_pred[0,v[0],v[1],0])))
            labelu = y_true[0, u[0], u[1], 0]
            labelv = y_true[0, v[0], v[1], 0]
            nlabels[u] = labelu
            nlabels[v] = labelv

        [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels)

        # for class imbalance
        posWeight = 0.0
        negWeight = 0.0
        # start off with every point in own cluster
        posError = totalPos
        negError = 0.0

        WY = np.zeros((1, 480, 320, 2), np.float32)
        #SY = np.ones((1, 480, 320, 2), np.float32)
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
                #if nlabels[u] != nlabels[v]:
                    #SY[0, u[0], u[1], channel] = -1.0
            if WS < 0.0: 
                WY[0, u[0], u[1], channel] = abs(WS) + negWeight
                #if nlabels[u] == nlabels[v]:
                    #SY[0, u[0], u[1], channel] = -1.0
        
        # Std normalization
        totalW = np.sum(WY)
        if totalW > 0.0:
            WY = WY / totalW

        return WY

    WY = tf.py_func(GetRandWeights, [y_true, y_pred], [tf.float32]) 
    
    #newY = tf.multiply(SY, gt_edges)

    edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(affs, gt_edges)))

    weightedLoss = tf.multiply(WY, edgeLoss)

    return tf.reduce_sum(weightedLoss) 

def weighted_rand_loss(input_nlabels=None):
    if input_nlabels == None:
        print('error in loss function')
        return
    
    #nlabels is expected to have shape (480, 320, 1)
    def rand_loss(y_true, y_pred):
        def GetRandWeights(nlabels):
            G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
            #nlabels = K.eval(input_nlabels)
            nlabels_dict = dict()
            for u, v, d in G.edges(data = True):
                if u[0] == v[0]:
                    channel = 0
                else:
                    channel = 1
                d['weight'] =  y_pred[0, u[0], u[1], channel]
                nlabels_dict[u] = (int) nlabels[0, u[0], u[1], 0]
                nlabels_dict[v] = (int) nlabels[0, v[0], v[1], 0]

            [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
            
            # for class imbalance
            posWeight = 0.0
            negWeight = 0.0
            # start off with every point in own cluster
            posError = totalPos
            negError = 0.0

            WY = np.zeros((1, 480, 320, 2), np.float32)
            #SY = np.ones((1, 480, 320, 2), np.float32)
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
                    #if nlabels[u] != nlabels[v]:
                        #SY[0, u[0], u[1], channel] = -1.0
                if WS < 0.0: 
                    WY[0, u[0], u[1], channel] = abs(WS) + negWeight
                    #if nlabels[u] == nlabels[v]:
                        #SY[0, u[0], u[1], channel] = -1.0
            
            # Std normalization
            totalW = np.sum(WY)
            if totalW > 0.0:
                WY = WY / totalW

            return WY
        
        WY = tf.py_func(GetRandWeights, [input_nlabels], [tf.float32])

        edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y_pred, y_true)))

        weightedLoss = tf.multiply(WY, edgeLoss)

        return tf.reduce_sum(weightedLoss)
    return rand_loss       

def unet(pretrained_weights=None):
    input_image = Input(shape=(480, 320, 3), name='images')
    input_nlabels = Input(shape=(480, 320, 1), name='nlabels')
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
  
    model = Model(input=[input_image, input_nlabels], output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = [weighted_rand_loss(input_nlabels)])

    

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    print("Init")
    model = unet()
    print("Exit")