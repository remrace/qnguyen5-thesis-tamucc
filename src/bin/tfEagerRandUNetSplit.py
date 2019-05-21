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

global lr

lr = 0.01
    
def get_lr():
    global lr
    lr = lr - (lr * 0.001)
    print('lr', lr)
    return lr    


INPUT_SIZE = (24, 24)

NUM_SAMPLES = 2
NUM_VALID = 1
BATCH_SIZE = 1
NUM_EPOCHS = 10000

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

def conv_block(input_tensor, num_filters):
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)    
    encoder = tf.keras.layers.Activation('relu')(encoder)
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)    
    encoder = tf.keras.layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)    
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)    
    decoder = tf.keras.layers.Activation('relu')(decoder)
    return decoder

####################################################################################################################################

def runExp():
    #tf.enable_eager_execution()
    print('Getting data')
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()  
    MAX_IND = 96
    # Training set
    im = list()
    gt = list()
    for i in range(NUM_SAMPLES):
        im.append(data[0][i][0:MAX_IND:4,0:MAX_IND:4,:])
        gt.append(data[1][i][0:MAX_IND:4,0:MAX_IND:4,:])
    X = np.float32(np.array(im))
    Y = np.float32(np.array(gt))

    # Validation set
    im = list()
    gt = list()
    for i in range(NUM_VALID):
        im.append(data[0][i+NUM_SAMPLES][0:MAX_IND:4,0:MAX_IND:4,:])
        gt.append(data[1][i+NUM_SAMPLES][0:MAX_IND:4,0:MAX_IND:4,:])
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
    unet = True
    if unet:
        input_image = tf.keras.layers.Input(shape=(24,24,3))
        
        #filters = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(input_image)  # input shape required
        #filters = tf.keras.layers.Activation('relu')(filters)
        #out = tf.keras.layers.Conv2D(1, (1,1))(filters)

        encoder0_pool, encoder0 = encoder_block(input_image, 16)
        center = conv_block(encoder0_pool, 32)
        decoder0 = decoder_block(center, encoder0, 16)
        out = tf.keras.layers.Conv2D(1, (1,1))(decoder0)
        
        model = tf.keras.Model(inputs=input_image, outputs=out)
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, (3, 3), padding='same', input_shape=(24, 24, 3))  # input shape required
        ])

    model.summary()
    #return
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

        PW = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 1), np.float32)
        NW = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 1), np.float32)
                    
        for i in range(len(posCounts)):
            posError = posError - posCounts[i]
            negError = negError + negCounts[i]
         
            (u,v) = mstEdges[i]

            PW[0, u[0], u[1], 0] = PW[0, u[0], u[1], 0] + (posError/2.0)
            PW[0, v[0], v[1], 0] = PW[0, v[0], v[1], 0] + (posError/2.0)
            NW[0, u[0], u[1], 0] = NW[0, u[0], u[1], 0] + (negError/2.0)
            NW[0, v[0], v[1], 0] = NW[0, v[0], v[1], 0] + (negError/2.0)

        WY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 1), np.float32)
        SY = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 1), np.float32)

        numPP = 0
        numPN = 0
        numWP = 0
        numWN = 0
        USE_DIFF = False

        for i in range(INPUT_SIZE[0]):
            for j in range(INPUT_SIZE[1]):
                if USE_DIFF:
                    diff = PW[0, i, j, 0] - NW[0, i, j, 0]
                    if diff > 0.0: 
                        SY[0, i, j, 0] = 1.0
                        WY[0, i, j, 0] = diff
                        numWN = numWN + NW[0, i, j, 0]
                    elif diff < 0.0:
                        SY[0, i, j, 0] = -1.0
                        WY[0, i, j, 0] = -diff
                        numWP = numWP + PW[0, i, j, 0]
                    else:
                        numWN = numWN + NW[0, i, j, 0]
                        numWP = numWP + PW[0, i, j, 0]
                else:
                    if YP[0, i, j, 0] > 0.0:
                        WY[0, i, j, 0] = NW[0, i, j, 0]
                        SY[0, i, j, 0] = -1.0
                        numPP = numPP + 1.0
                        if NW[0, i, j, 0] > 0.1:
                            numWN = numWN + 1
                    elif YP[0, i, j, 0] < 0.0:
                        WY[0, i, j, 0] = PW[0, i, j, 0]
                        SY[0, i, j, 0] = 1.0                
                        numPN = numPN + 1.0
                        if PW[0, i, j, 0] > 0.1:
                            numWP = numWP + 1
        
        # Std normalization        
        totalW = np.sum(WY)
        if totalW != 0.0:
            WY = np.divide(WY, totalW)        
        
        if USE_DIFF:
            print("W: " + str(totalW) + " Ignore: " + str(numWN+numWP) + " from Pos: " + str(numWP) + " and " + str(numWN))
        else:
            print("W: " + str(totalW) + " Pos: " + str(numWN) + " of " + str(numPP) + " Neg: " + str(numWP) + " of " + str(numPN) + " from " + str(numPP + numPN) + " out of " + str(INPUT_SIZE[0] * INPUT_SIZE[1]))

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

    global_step = tf.Variable(0)    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=get_lr)    

    for epoch in range(NUM_EPOCHS):        
        rp = randint(0, 1)
        #rp = epoch % 2
        #rp = 0
        XX = X[rp:rp+1,:,:,:]
        YY = Y[rp:rp+1,:,:,:]
        YL = YY[0,:,:,0]

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
