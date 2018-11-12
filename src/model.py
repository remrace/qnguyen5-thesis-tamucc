import tensorflow as tf
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import networkx as nx
import SegEval as ev

import tensorflow.contrib as tfcontrib
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

INPUT_SIZE = (481, 321, 3)
NUM_OUTPUTS = 1
KERNEL_SIZE = 5
N = (INPUT_SIZE[0]-1) * INPUT_SIZE[1] + (INPUT_SIZE[1]-1) * INPUT_SIZE[0]
D = KERNEL_SIZE * KERNEL_SIZE * 3  

'''
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
                #if nlabels_dict[u] != nlabels_dict[v]:
                    #SY[0, u[0], u[1], channel] = -1.0
                SY[0, u[0], u[1], channel] = 1.0
            if WS < 0.0: 
                WY[0, u[0], u[1], channel] = abs(WS) + negWeight
                #if nlabels_dict[u] == nlabels_dict[v]:
                    #SY[0, u[0], u[1], channel] = -1.0
                SY[0, u[0], u[1], channel] = -1.0
        
        # Std normalization
        totalW = np.sum(WY)
        WY = np.divide(WY, totalW)

        return WY, SY
    WY, SY = tf.py_func(GetRandWeights, [nlabels, y_pred], [tf.float32, tf.float32])
    return [WY, SY]
'''

def rand_weight(y_pred, nlabels):
    def GetRandWeights(y, n):
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        edgeInd = dict()
        upto = 0
        for u, v, d in G.edges(data = True):
            d['weight'] = y[0, upto]
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

        WY = np.zeros( (1, N, 1), np.float32)
        SY = np.ones( (1, N, 1), np.float32)
        for i in range(len(posCounts)):
            posError = posError - posCounts[i]
            negError = negError + negCounts[i]

            WS = posError - negError
            
            ind = edgeInd[ mstEdges[i] ]
            WY[0, ind] = abs(WS)

            if u[0] == v[0]: #vertical, dy, channel 0
                channel = 0
            else: #horizontal, dx, channel 1
                channel = 1
            
            if WS > 0.0:
                if nlabels_dict[u] != nlabels_dict[v]:
                    SY[0, ind] = -1.0
                
            if WS < 0.0: 
                if nlabels_dict[u] == nlabels_dict[v]:
                    SY[0, ind] = -1.0
        
        # Std normalization
        totalW = np.sum(WY)
        WY = WY / totalW

        return WY, SY
    WY, SY = tf.py_func(GetRandWeights, [y_pred, nlabels], [tf.float32, tf.float32])
    return [WY, SY]

def output_shape(input_shape):
	return [input_shape, input_shape]


def ff_loss(WY, SY):
    def f_loss(y_true, y_pred):
        newY = tf.maximum(SY, y_true)

        edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y_pred, newY)))

        weightedLoss = tf.multiply(WY, edgeLoss)

        return tf.reduce_sum(weightedLoss)
    return f_loss

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def image_to_vect

def unet(pretrained_weights=None):
    input_image = layers.Input(shape=(N,D), name='input_image')
    input_nlabels = layers.Input(shape=(481, 321, 1), name='input_nlabels')
    
    encoder0_pool, encoder0 = encoder_block(input_image, 32)
    # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 8
    center = conv_block(encoder4_pool, 1024)
    # center
    decoder4 = decoder_block(center, encoder4, 512)
    # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    outputs = layers.Conv2D(1, (1, 1), activation='relu')(decoder0)
    #scale conv10
    
    aff = layers.Conv2D(2, (2,2), padding='valid')(outputs)
    flat_aff = layers.Flatten()(aff)

    WY, SY = layers.Lambda(rand_weight, output_shape=output_shape, arguments={'nlabels': input_nlabels})(flat_aff)

    model = models.Model(inputs=[input_image, input_nlabels], outputs = [aff])
    model.compile(optimizer = SGD(lr=0.1, momentum=0.5, decay=0.0, nesterov=True), loss = ff_loss(WY, SY))

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    print("Init")
    model = unet()
    print("Exit")