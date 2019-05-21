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
NUM_SAMPLES = 5
BATCH_SIZE = 1
NUM_EPOCHS = 200
MAX_IDX = 96
REDUCTION = 4
KERNEL_SIZE = 3
SIZE = (MAX_IDX//REDUCTION, MAX_IDX//REDUCTION, 3)
LR = 0.05
N_size = 5
pad = (N_size-1)//2
D = N_size*N_size*3
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
def _save(im1, im2, fname):
    fig=plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(im1, cmap='tab20')
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(im2, cmap='tab20')
    plt.axis('off')
    fig.savefig(fname)

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
        X = np.zeros((NUM_SAMPLES, SIZE[0]*(SIZE[0]-1)*2, 3), dtype=np.single)
        G = nx.grid_2d_graph(SIZE[0], SIZE[1])
        for i in range(imgs.shape[0]): #for each sample      
            aimg = np.zeros((SIZE[0]*(SIZE[0]-1)*2, 3))
            for j in range(imgs.shape[-1]): #for each channel
                img = imgs[i, :, :, j]
                img = signal.convolve2d(img, np.ones((KERNEL_SIZE, KERNEL_SIZE))/9.0, mode='same', boundary='symm')
                upto = 0
                for (u,v) in G.edges():
                    aimg[upto][j] = (img[u]+img[v])/2
                    upto = upto + 1
            X[i,:,:] = aimg
        return X

    def feature_maker2(imgs):
        #expected 4D arrays NxHxWxC
        X = []
        G = nx.grid_2d_graph(SIZE[0], SIZE[1])
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :]
            sobel_x = np.c_[
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
            ]

            sobel_y = np.c_[
                [1,2,1],
                [0,0,0],
                [-1,-2,-1]
            ]

            for b in range(img.shape[-1]):
                sx = signal.convolve2d(img[:,:,b], sobel_x, mode="same", boundary="symm")
                sy = signal.convolve2d(img[:,:,b], sobel_y, mode="same", boundary="symm")
                temp = np.sqrt(sx*sx + sy*sy)
                img[:,:,b] = temp
                
            a = np.pad(img, pad_width=((pad,pad),(pad,pad),(0,0)), mode='constant')
            upto=0
            e = []
            for (u,v) in G.edges():
                N1 = a[u[0]:u[0]+N_size,u[1]:u[1]+N_size,:]
                N2 = a[v[0]:v[0]+N_size,v[1]:v[1]+N_size,:]
                N = (N1+N2)/2.0
                upto = upto + 1
                e.append(N.flatten())
            X.append(np.array(e))
        return np.array(X)

    X = feature_maker2(X)
    XT = feature_maker2(XT)
    print('Training set')
    print(X.shape)
    print(Y.shape)
    print('Testing set')
    print(XT.shape)
    print(YT.shape)
    return X, Y, XT, YT

def rand_image(YP):
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    upto = 0
    for u, v, d in G.edges(data = True):
        d['weight'] =  YP[upto, 0]
        upto = upto + 1
    L = ev.GetLabelsAtThreshold(G, 0.0)
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
        d['weight'] =  YP[upto, 0]
        upto = upto + 1
        nlabels_dict[u] = Y[u[0], u[1], 0]
        nlabels_dict[v] = Y[v[0], v[1], 0]
    error = ev.FindRandErrorAtThreshold(G, nlabels_dict, 0.0)
    return error

def get_rand_weight(YP, Y):
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    nlabels_dict = dict()
    edge_idx = dict()
    upto = 0
    for u, v, d in G.edges(data = True):
        d['weight'] =  YP[0, upto, 0]
        edge_idx[(u,v)] = upto
        upto = upto + 1
        nlabels_dict[u] = Y[0, u[0], u[1], 0]
        nlabels_dict[v] = Y[0, v[0], v[1], 0]
    [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
    posError = totalPos
    negError = 0.0
    WY = np.zeros((1, SIZE[0]*(SIZE[0]-1)*2, 1), np.single)
    SY = np.zeros((1, SIZE[0]*(SIZE[0]-1)*2, 1), np.single)
    for i in range(len(posCounts)):
        e = edge_idx[mstEdges[i]]
        #w = (len(posCounts)-i)/len(posCounts)
        '''
        if YP[0, e, 0] < 0.0:
            WY[0, e, 0] = posCounts[i]/(posCounts[i]+negCounts[i])
            SY[0, e, 0] = 1.0
        elif YP[0, e, 0] > 0.0:
            WY[0, e, 0] = negCounts[i]/(posCounts[i]+negCounts[i])
            SY[0, e, 0] = -1.0
        '''
        WS = posCounts[i] - negCounts[i]
        if WS > 0.0:
            SY[0, e, 0] = 1.0
            WY[0, e, 0] = WS/(posCounts[i]+negCounts[i])
        elif WS < 0.0:
            SY[0, e, 0] = -1.0
            WY[0, e, 0] = -WS/(posCounts[i]+negCounts[i])
        
    return [WY, SY]

def runExp(): 
    X_train, Y_train, XT_test, YT_test = getData()
    print('## Model ')
    #input_edge = tf.keras.layers.Input(shape=(SIZE[0]*(SIZE[0]-1)*2, SIZE[2]))
    input_edge = tf.keras.layers.Input(shape=(SIZE[0]*(SIZE[0]-1)*2, D))
    out = tf.keras.layers.LocallyConnected1D(1,1)(input_edge)
    model = tf.keras.Model(inputs=input_edge, outputs=out)
    model.summary()

    def myloss(YP, Y):
        WY, SY = get_rand_weight(YP, Y)
        loss = tf.losses.hinge_loss(labels=SY, logits=YP, weights=WY, reduction=tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)
        return loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
    #optimizer = tf.train.AdamOptimizer()
    loss_kruskal = []
    val_loss_kruskal = []
    rand_kruskal = []
    val_rand_kruskal =[]
    minR = 1.0
    for epoch in range(NUM_EPOCHS):
        #WY, SY = get_rand_weight(model(X),Y)
        loss_avg = []
        val_loss_avg = []
        rand_avg = []
        val_rand_avg = []
        for i in range(X_train.shape[0]):
            X = np.expand_dims(X_train[i,:], axis=0)
            Y = np.expand_dims(Y_train[i,:], axis=0)
            with tf.GradientTape() as tape:
                YP = model(X)
                loss = myloss(YP, Y)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

        if (epoch+1) % 5 == 0:
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
                #v_pred_img = rand_image(YPT[0]) 
                #debugImages(YT[0,:,:,0], v_pred_img)
                if i%2==0:
                    v_pred_img = rand_image(YPT[0])
                    _save(YT[0,:,:,0], v_pred_img, fname='linear/diff_'+str(LR)+'_epoch_'+'_'+str(epoch+1)+'_'+str(i)+'.png')
                v_err = rand_error(YPT[0], YT[0])
                val_loss_avg.append(v_loss.numpy())
                val_rand_avg.append(v_err)

            loss_kruskal.append(np.average(loss_avg))
            val_loss_kruskal.append(np.average(val_loss_avg))
            rand_kruskal.append(np.average(rand_avg))
            val_rand_kruskal.append(np.average(val_rand_avg))
            if val_rand_kruskal[-1] < minR:
                model.save_weights('linear/diff_'+ str(LR)+ '_best' + str(epoch+1) + '_' + str(val_rand_kruskal[-1]) + '.h5')
                minR = val_rand_kruskal[-1]
            print("Epoch: {},   Loss: {}    Err: {}     Val Loss: {}    Val Err: {}".format(epoch+1, np.average(loss_avg), np.average(rand_avg), np.average(val_loss_avg), np.average(val_rand_avg)))
    del model
    with open('linear/diff_'+ str(LR)+ '.log', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss_kruskal', 'val_loss_kruskal', 'rand_kruskal', 'val_rand_kruskal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(loss_kruskal)):
            writer.writerow({'epoch':(i+1)*5, 
                            'loss_kruskal':loss_kruskal[i], 
                            'val_loss_kruskal':val_loss_kruskal[i], 
                            'rand_kruskal':rand_kruskal[i], 
                            'val_rand_kruskal':val_rand_kruskal[i]})
    

####################################################################################################################################
####################################################################################################################################


if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    runExp()
    #getData()