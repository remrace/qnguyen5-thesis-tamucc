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
IMAGE_SIZE = (100, 100, 3)
INPUT_SIZE = (100, 100, 3)



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

        WY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)
        SY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)
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

def rand_weight_inference(y_pred, nlabels):
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

        [bestT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
        
        # for class imbalance
        posWeight = 0.0
        negWeight = 0.0
        # start off with every point in own cluster
        posError = totalPos
        negError = 0.0

        WY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)
        SY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)
        TY = bestT*np.ones((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)
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

        return WY, SY, TY
    WY, SY, TY = tf.py_func(GetRandWeights, [nlabels, y_pred], [tf.float32, tf.float32, tf.float32])
    return [WY, SY, TY]

def sobel_edge(image):
    out = tf.image.sobel_edges(image)
    shape = tf.shape(out)
    return tf.reshape(out, [shape[0], shape[1], shape[2], -1])

def sobel_output_shape(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], 6)

def randweight_output_shape(input_shape):
	return [input_shape, input_shape]

def randweight_output_shape_inference(input_shape):
	return [input_shape, input_shape, input_shape]

def ff_loss(WY, SY, TY = None):
    SY = tf.reshape(SY, [-1])
    WY = tf.reshape(WY, [-1])
    if TY is not None:
        TY = tf.reshape(TY, [-1])
    def f_loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        #newY = tf.maximum(SY, y_true)
        if TY is not None:
            edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(tf.subtract(y_pred, TY), SY)))
        else:
            edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y_pred, SY)))

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

#input_edge = layers.Lambda(sobel_edge, output_shape=sobel_output_shape)(input_image)
def unet(USE_CC_INFERENCE=None, pretrained_weights=None):
    input_image = layers.Input(shape=(IMAGE_SIZE), name='input_image')
    input_nlabels = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1],1), name='input_nlabels')

    input_edges = layers.Lambda(sobel_edge, output_shape=sobel_output_shape)(input_image)

    encoder0_pool, encoder0 = encoder_block(input_edges, 64)

    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 128)

    #encoder2_pool, encoder2 = encoder_block(encoder1_pool, 256)

    #encoder3_pool, encoder3 = encoder_block(encoder2_pool, 512)
 
    center = conv_block(encoder1_pool, 256)

    #decoder3 = decoder_block(center, encoder3, 512)

    #decoder2 = decoder_block(decoder3, encoder2, 256)

    decoder1 = decoder_block(center, encoder1, 128)
 
    decoder0 = decoder_block(decoder1, encoder0, 64)

    #decoder0 = layers.Cropping2D(18)(decoder0)

    aff = layers.Conv2D(2, (1, 1))(decoder0)
    aff = layers.Activation('tanh')(aff)
    #aff = layers.Lambda(sobel_edge, output_shape=sobel_output_shape)(enhanced_image)
    
    if USE_CC_INFERENCE:
        print("USE_CC_INFERENCE: YES")
        WY, SY, TY = layers.Lambda(rand_weight_inference, output_shape=randweight_output_shape_inference, arguments={'nlabels': input_nlabels})(aff)
        model = models.Model(inputs=[input_image, input_nlabels], outputs = [aff])
        model.compile(optimizer = SGD(lr=0.1), loss = ff_loss(WY, SY, TY))
    else:
        print("USE_CC_INFERENCE: NO")
        WY, SY = layers.Lambda(rand_weight, output_shape=randweight_output_shape, arguments={'nlabels': input_nlabels})(aff)
        model = models.Model(inputs=[input_image, input_nlabels], outputs = [aff])
        model.compile(optimizer = SGD(lr=0.1), loss = ff_loss(WY, SY))


    model.summary()
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    print("Init")
    model = unet()
    print("Exit")