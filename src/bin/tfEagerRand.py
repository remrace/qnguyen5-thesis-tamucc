from __future__ import absolute_import, division, print_function

import numpy as np 
np.random.seed(1337) # for reproducibility

import tensorflow as tf
tf.enable_eager_execution()

import os
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import SegEval as ev

INPUT_SIZE = (25, 25)
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

if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    #tf.enable_eager_execution()
    print('Getting data')
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()
    print('Done getting data')

    NUM_SAMPLES = 10
    BATCH_SIZE = 1
    NUM_EPOCHS = 10

    im = list()
    gt = list()
    for i in range(NUM_SAMPLES):
        im.append(data[0][i][0:100:4,0:100:4,:])
        gt.append(data[1][i][0:100:4,0:100:4,:])
    
    X = np.float32(np.array(im))
    Y = np.float32(np.array(gt))
    flabels = np.float32(np.ones([NUM_SAMPLES, 25, 25, 2]))
    #flabels = np.ones([NUM_SAMPLES, 1, 1, 2])
    print(X.shape)
    print(X.dtype)
    print(Y.shape)
    print(flabels.shape)
    print('Start training')

    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (5, 5), input_shape=(25, 25, 3)),  # input shape required
    tf.keras.layers.Lambda(tf.image.sobel_edges)
    ])
    model.summary()
    #predictions = model(X)
    #print(predictions.shape)

    def rand_error(model, X, YN):        
        YP = model(X)        
        G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
        nlabels_dict = dict()
        for u, v, d in G.edges(data = True):
            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1
            d['weight'] =  YP[0, u[0], u[1], channel]
            nlabels_dict[u] = YN[0, u[0], u[1], 0]
            nlabels_dict[v] = YN[0, v[0], v[1], 0]    
        error = ev.FindRandErrorAtThreshold(G, nlabels_dict, 0.0)
        return error

    def rand_loss(model, X, YN):        
        def get_rand_weight(YP, YN):
            G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
            nlabels_dict = dict()
            for u, v, d in G.edges(data = True):
                if u[0] == v[0]:
                    channel = 0
                else:
                    channel = 1
                d['weight'] =  YP[0, u[0], u[1], channel]
                nlabels_dict[u] = YN[0, u[0], u[1], 0]
                nlabels_dict[v] = YN[0, v[0], v[1], 0]

            [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
            
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
                    WY[0, u[0], u[1], channel] = abs(WS)
                    SY[0, u[0], u[1], channel] = 1.0
                if WS < 0.0: 
                    WY[0, u[0], u[1], channel] = abs(WS)
                    SY[0, u[0], u[1], channel] = -1.0
            
            # Std normalization
            totalW = np.sum(WY)
            if totalW != 0.0:
                WY = np.divide(WY, totalW)

            return [WY, SY]

        YP = model(X)        
        #WY, SY = tf.py_func(get_rand_weight, [YP, YN], [tf.float32, tf.float32])
        WY, SY = get_rand_weight(YP, YN) 
        #newY = tf.multiply(SY, YT)        
        edgeLoss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(YP, SY)))        
        weightedLoss = tf.multiply(WY, edgeLoss)

        return tf.reduce_sum(weightedLoss)

    ##################################################################
    ########################
    XX = X[0:BATCH_SIZE,:,:,:]
    YY = Y[0:BATCH_SIZE,:,:,:]
    print(XX.shape)    
    print(YY.shape)



    l = rand_loss(model, XX, YY)
    print("Loss test: {}".format(l))

    e = rand_error(model, XX, YY)
    print("Error test: {}".format(e))
    
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = rand_loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    train_loss_results = []
    train_accuracy_results = []
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.contrib.eager.Variable(0)    

    loss_value, grads = grad(model, XX, YY) 

    print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                           loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          rand_loss(model, XX, YY).numpy()))
    e = rand_error(model, XX, YY)
    print("Error test: {}".format(e))


    #for epoch in range(NUM_EPOCHS):
    #    epoch_loss_avg = tfe.metrics.Mean()
    #    epoch_accuracy = tfe.metrics.Accuracy()

    #    # Training loop - using batches of 1
    #    for ti in range(X.shape[0]):
    #        # Optimize the model
    #        loss_value, grads = grad(model, X, y)
    #optimizer.apply_gradients(zip(grads, model.trainable_variables),
    #                          global_step)

    # Track progress
    #epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    #epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    #train_loss_results.append(epoch_loss_avg.result())
    #train_accuracy_results.append(epoch_accuracy.result())
  
    #if epoch % 50 == 0:
    #print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
    #                                                            epoch_loss_avg.result(),
    #                                                            epoch_accuracy.result()))    
