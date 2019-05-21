from __future__ import absolute_import, division, print_function

import numpy as np 
np.random.seed(1337) # for reproducibility

import tensorflow as tf
tf.enable_eager_execution()
from random import randint
import os
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import SegEval as ev

INPUT_SIZE = (25, 25)

NUM_SAMPLES = 1
NUM_VALID = 1
BATCH_SIZE = 1
NUM_EPOCHS = 1000

def debugImages(im1, im2):
    pic = dict()
    pic[-1] = 'x'
    pic[0] = '.'
    pic[1] = '+'
    pic[2] = '*'
    pic[3] = '#'
    pic[4] = '$'
    pic[5] = '%'
    pic[6] = '?'
    print("debug:")
    for i in range(im1.shape[0]):
        line1 = ""
        line2 = ""
        for j in range(im1.shape[1]):
            id1 = int(im1[i,j])
            if id1 > 0: 
                id1 = id1 % 7
            line1 = line1 + pic[id1]
            id2 = int(im2[i,j])
            if id2 > 0: 
                id2 = id2 % 7
            line2 = line2 + pic[id2]
        line = line1 + '   :   ' + line2
        print(line)  

####################################################################################################################################
####################################################################################################################################

def runExp():
    #tf.enable_eager_execution()
    print('Getting data')
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()  
        
    # Training set
    im = list()
    gt = list()
    for i in range(NUM_SAMPLES):
        im.append(data[0][i][0:100:4,0:100:4,:])
        gt.append(data[1][i][0:100:4,0:100:4,:])
    X = np.float32(np.array(im))
    Y = np.float32(np.array(gt))

    # Validation set
    im = list()
    gt = list()
    for i in range(NUM_VALID):
        im.append(data[0][i+NUM_SAMPLES][0:100:4,0:100:4,:])
        gt.append(data[1][i+NUM_SAMPLES][0:100:4,0:100:4,:])
    XT = np.float32(np.array(im))
    YT = np.float32(np.array(gt))
    
    print('Training set')
    print(X.shape)
    print(Y.shape)
    
    print('Testing set')
    print(XT.shape)
    print(YT.shape)

    print('#########################################')
    print('## Model ')
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (1, 1), input_shape=(25, 25, 3))  # input shape required
    ])
    model.summary()
    #predictions = model(X)
    #print(predictions.shape)

    ####################################################################################################################################
    ####################################################################################################################################
    print('#########################################')
    print('## Training ')

    def rand_image(YP):
        
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        for u, v, d in G.edges(data = True):
            d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0

        L = ev.GetLabelsAtThreshold(G)
        img = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1]), np.float32)
        for i in range(img.shape[0]):        
            for j in range(img.shape[1]):                
                img[i,j] = L[(i,j)]

        return img

    def rand_error(YP, YN):                
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        for u, v, d in G.edges(data = True):
            d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0
            nlabels_dict[u] = YN[0, u[0], u[1], 0]
            nlabels_dict[v] = YN[0, v[0], v[1], 0]    

        error = ev.FindRandErrorAtThreshold(G, nlabels_dict, 0.0)
        return error

    def get_rand_weight(YP, YN):
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        for u, v, d in G.edges(data = True):
                        
            d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0
            nlabels_dict[u] = YN[0, u[0], u[1], 0]
            nlabels_dict[v] = YN[0, v[0], v[1], 0]

        [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
        
        # start off with every point in own cluster
        posError = totalPos
        negError = 0.0

        WY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 1), np.float32)
        SY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 1), np.float32)
                    
        for i in range(len(posCounts)):
            posError = posError - posCounts[i]
            negError = negError + negCounts[i]

            WS = posError - negError


            (u,v) = mstEdges[i]

            WY[0, u[0], u[1], 0] = abs(WS)/2.0
            WY[0, v[0], v[1], 0] = abs(WS)/2.0
            if WS > 0.0:                
                SY[0, u[0], u[1], 0] = 1.0                                    
                SY[0, v[0], v[1], 0] = 1.0                    
            if WS < 0.0: 
                SY[0, u[0], u[1], 0] = -1.0                                    
                SY[0, v[0], v[1], 0] = -1.0                    
        
        # Std normalization
        totalW = np.sum(WY)
        if totalW != 0.0:
            WY = np.divide(WY, totalW)
        
        return [WY, SY]
        
    ####################################################################################################################################
    ####################################################################################################################################

    XX = X[0:BATCH_SIZE,:,:,:]
    YY = Y[0:BATCH_SIZE,:,:,:]
    #YYL = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)

    YL = YY[0,:,:,0]
    
    YP = model(XX)

    img = rand_image(YP)
    debugImages(YL, img)

    #l = test_loss(YP, YYL)    
    #e = rand_error(model, XX, YY)
    #print("Train Loss: {} Error: {}".format(l,e))

    #lt = rand_loss(model, XXT, YYT)
    #et = rand_error(model, XXT, YYT)
    #print("Test  Loss: {} Error: {}".format(lt,et))    
       
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer = tf.train.AdadeltaOptimizer()
    global_step = tf.contrib.eager.Variable(0)    

    for epoch in range(NUM_EPOCHS):
        #rp = randint(0, NUM_SAMPLES-1)
        rp = 0
        #XX = X[rp:rp+1,:,:,:]
        #YY = Y[rp:rp+1,:,:,:]
        #YYL = np.ones((1, INPUT_SIZE[0], INPUT_SIZE[1], 2), np.float32)
        YPT = model(XX)
        WY, SY = get_rand_weight(YPT, YY) 
                
        with tf.GradientTape() as tape:
            YP = model(XX)
            #LF = tf.square(YP - SY)
            LF = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(YP, SY)))        
            lossVals = tf.multiply(LF, WY)

        grads = tape.gradient(lossVals, model.trainable_variables)
        #print("===============>")
        #print(loss_value)
        #print(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
        
        img = rand_image(YP)
        debugImages(YL, img)
        
        myLoss = tf.reduce_sum(lossVals)

        err = rand_error(YP, YY)

        print("Step: {},         Loss: {}     Err: {}".format(global_step.numpy(), myLoss, err))

        #e = rand_error(model, XX, YY)
        
        #print("Step: {},      Loss: {}   Error: {}".format(global_step.numpy(), loss_value, e))
        #lt = rand_loss(model, XXT, YYT)
        #et = rand_error(model, XXT, YYT)
        #print("      Test        Loss: {}          Error: {}".format(lt,et))    

    #XX = X[0:1,:,:,:]
    #YY = Y[0:1,:,:,:]
    #yimg = Y[0,:,:,0]
    
    #et = rand_error(model, XX, YY)
    #print("Final error: " + str(et))

    #img = generate_image(model, XX)
    #debugImages(yimg, img) 
    

if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    runExp()
