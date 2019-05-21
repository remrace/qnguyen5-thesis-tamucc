from __future__ import absolute_import, division, print_function

import numpy as np
np.random.seed(1337) # for reproducibility
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
tf.enable_eager_execution()
from scipy import signal
import os
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import SegEval as ev
from random import randint
import networkx as nx
import csv
from statistics import mean

NUM_SAMPLES = 30
BATCH_SIZE = 1
NUM_EPOCHS = 500
MAX_IDX = 96
REDUCTION = 4
KERNEL_SIZE = 3
SIZE = (MAX_IDX//REDUCTION, MAX_IDX//REDUCTION, 3)
NUM_EXP = 1

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

###########################################################################################################
def getData():
    # Training set
    im = list()
    gt = list()
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()  
    for i in range(NUM_SAMPLES):
        im.append(data[0][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:])
        gt.append(data[1][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:])
    X = np.array(im).astype(np.single)
    Y = np.array(gt).astype(np.single)
    # Validation set
    im = list()
    gt = list()
    f = open('../synimage/test.p', 'rb')
    data = pickle.load(f)
    f.close() 
    for i in range(NUM_SAMPLES):
        im.append(data[0][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:])
        gt.append(data[1][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:])
    XT = np.array(im).astype(np.single)
    YT = np.array(gt).astype(np.single)

    def feature_maker(imgs):
        #expected 4D arrays NxHxWxC
        for i in range(imgs.shape[0]): #for each sample      
            for j in range(imgs.shape[-1]): #for each channel
                img = imgs[i, :, :, j]
                imgs[i,:,:,j] = signal.convolve2d(img, np.ones((KERNEL_SIZE, KERNEL_SIZE))/9.0, mode='same', boundary='symm')
        return imgs

    X = feature_maker(X)
    XT = feature_maker(XT)

    print('Training set')
    print(X.shape)
    print(Y.shape)
    print('Testing set')
    print(XT.shape)
    print(YT.shape)
    return X, Y, XT, YT

def rand_image(YP):
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1
        d['weight'] =  YP[u[0], u[1], channel]
    [bestT, bestE] = ev.FindMinEnergyThreshold(G)
    L = ev.GetLabelsAtThreshold(G, bestT)
    img = np.zeros((SIZE[0], SIZE[1]), np.single)
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            img[i,j] = L[(i,j)]
    return img

def rand_error(YP, Y):              
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    nlabels_dict = dict()
    upto = 0
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1
        d['weight'] =  YP[u[0], u[1], channel]
        nlabels_dict[u] = Y[u[0], u[1], 0]
        nlabels_dict[v] = Y[v[0], v[1], 0]
    [bestT, bestE] = ev.FindMinEnergyThreshold(G)
    error = ev.FindRandErrorAtThreshold(G, nlabels_dict, bestT)
    return error

def get_rand_weight(YP, Y):
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1
        d['weight'] =  YP[0, u[0], u[1], channel]
        nlabels_dict[u] = Y[0, u[0], u[1], 0]
        nlabels_dict[v] = Y[0, v[0], v[1], 0]
    #[posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
    [lowT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
    WY = np.zeros((1, SIZE[0], SIZE[1], 2), np.single)
    SY = np.zeros((1, SIZE[0], SIZE[1], 2), np.single)

    for i in range(len(posCounts)):
        (u,v) = mstEdges[i]
        if u[0] == v[0]: #vertical, dy, channel 0
            channel = 0
        else: #horizontal, dx, channel 1
            channel = 1
        WS = posCounts[i] - negCounts[i]
        if WS > 0.0:
            SY[0, u[0], u[1], channel] = 1.0
            WY[0, u[0], u[1], channel] = WS/(posCounts[i]+negCounts[i])
        elif WS < 0.0:
            SY[0, u[0], u[1], channel] = -1.0
            WY[0, u[0], u[1], channel] = -WS/(posCounts[i]+negCounts[i])
    return [WY, SY, lowT]

def _save(im1, im2, fname):
    fig=plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(im1, cmap='tab20')
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(im2, cmap='tab20')
    plt.axis('off')
    fig.savefig(fname)

def runExp(id=0): 
    X_train, Y_train, XT_test, YT_test = getData()
    print('## Model ')
    input_image = tf.keras.layers.Input(shape=SIZE)
    encoder0_pool, encoder0 = encoder_block(input_image, 16)
    center = conv_block(encoder0_pool, 32)
    decoder0 = decoder_block(center, encoder0, 16)
    #out = tf.keras.layers.Conv2D(2, (1,1))(decoder0)
    
    model = tf.keras.Model(inputs=input_image, outputs=out)
    model.summary()

    def myloss(YP, Y):
        WY, SY, lowT = get_rand_weight(YP, Y)
        YP = tf.subtract(YP, lowT)
        loss = tf.losses.hinge_loss(labels=SY, logits=YP, weights=WY, reduction=tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)
        return loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

    loss_kruskal = []
    val_loss_kruskal = []
    rand_kruskal = []
    val_rand_kruskal =[]

    for epoch in range(NUM_EPOCHS):
        #WY, SY = get_rand_weight(model(X),Y)
        loss_avg = []
        val_loss_avg = []
        rand_avg = []
        val_rand_avg = []
        for i in range(X_train.shape[0]):
            X = np.expand_dims(X_train[i,:], axis=0)
            Y = np.expand_dims(Y_train[i,:], axis=0)
            WY, SY, lowT = get_rand_weight(model(X), Y)
            with tf.GradientTape() as tape:
                YP = model(X)
                YPL = tf.subtract(YP, lowT)
                loss = tf.losses.hinge_loss(labels=SY, logits=YPL, weights=WY, reduction=tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

        if (epoch+1) % 10 == 0:
            for i in range(X_train.shape[0]):
                #print('Train')
                X = np.expand_dims(X_train[i,:], axis=0)
                Y = np.expand_dims(Y_train[i,:], axis=0)
                YP = model(X)
                loss = myloss(YP, Y)
                #pred_img = rand_image(YP[0])
                #debugImages(Y[0,:,:,0], pred_img)
                err = rand_error(YP[0], Y[0])
                loss_avg.append(loss.numpy())
                rand_avg.append(err)
            for i in range(Y_train.shape[0]):
                #print('Test')
                XT = np.expand_dims(XT_test[i,:], axis=0)
                YT = np.expand_dims(YT_test[i,:], axis=0)
                YPT = model(XT)
                v_loss = myloss(YPT, YT)
                if i%5==0:
                    v_pred_img = rand_image(YPT[0]) 
                #    debugImages(YT[0,:,:,0], v_pred_img)
                    _save(YT[0,:,:,0], v_pred_img, fname='03-15/epoch'+'_'+str(epoch+1)+'_'+str(i))
                v_err = rand_error(YPT[0], YT[0])
                val_loss_avg.append(v_loss.numpy())
                val_rand_avg.append(v_err)
            
            loss_kruskal.append(np.average(loss_avg))
            val_loss_kruskal.append(np.average(val_loss_avg))
            rand_kruskal.append(np.average(rand_avg))
            val_rand_kruskal.append(np.average(val_rand_avg))
            #model.save_weights('03-15/2D_diff_0.05_' + str(epoch+1) + '_' + str(val_rand_kruskal[-1]) + '.h5')
            print("Epoch: {},   Loss: {}    Err: {}     Val Loss: {}    Val Err: {}".format(epoch+1, np.average(loss_avg), np.average(rand_avg), np.average(val_loss_avg), np.average(val_rand_avg)))
    del model
    with open('03-15/2D_diff_0.05_' + str(id) + '.log', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'val_loss', 'rand', 'val_rand']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(loss_kruskal)):
            writer.writerow({'epoch': (i+1)*10,
                            'loss': loss_kruskal[i],
                            'val_loss': val_loss_kruskal[i],
                            'rand': rand_kruskal[i],
                            'val_rand': val_rand_kruskal[i]})
    

####################################################################################################################################
####################################################################################################################################

def model_eval(fname):
    assert fname != None
    X_train, Y_train, XT_test, YT_test = getData()
    print('## Model ')
    input_image = tf.keras.layers.Input(shape=SIZE)
    encoder0_pool, encoder0 = encoder_block(input_image, 16)
    center = conv_block(encoder0_pool, 32)
    decoder0 = decoder_block(center, encoder0, 16)
    out = tf.keras.layers.Conv2D(2, (1,1))(decoder0)
    
    model = tf.keras.Model(inputs=input_image, outputs=out)
    model.load_weights(fname)

    affinities = []
    segmentations = []
    val_rand_avg = []
    for i in range(Y_train.shape[0]):
        XT = np.expand_dims(XT_test[i,:], axis=0)
        YT = np.expand_dims(YT_test[i,:], axis=0)
        YPT = model(XT)
        v_pred_img = rand_image(YPT[0]) 
        #_save(YT[0,:,:,0], v_pred_img, fname='03-15/epoch'+'_'+str(i))
        #plt.imsave(fnam='03-15/'+str(i)+'.png', v_pred_img)
        v_err = rand_error(YPT[0], YT[0])
        val_rand_avg.append(v_err)
        print('val_rand: ', np.average(val_rand_avg))

if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    #for i in range(NUM_EXP):
    #    runExp(id=i+4)
    #getData()
    #model_eval('03-15/'+ '2D_diff_0.05_80_0.1417020933977456.h5')