from __future__ import absolute_import, division, print_function

import numpy as np
np.random.seed(1337) # for reproducibility
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
tf.enable_eager_execution()

import os
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import SegEval as ev
from random import randint
import networkx as nx
import csv
SIZE = (50, 50, 3)
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

def getData():
    # Training set
    im = list()
    gt = list()
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()  
    for i in range(NUM_SAMPLES):
        im.append(data[0][i][0:100:2,0:100:2,:])
        gt.append(data[1][i][0:100:2,0:100:2,:])
    X = np.array(im).astype(np.single)
    Y = np.array(gt).astype(np.single)
    # Validation set
    im = list()
    gt = list()
    f = open('../synimage/test.p', 'rb')
    data = pickle.load(f)
    f.close() 
    for i in range(NUM_VALID):
        im.append(data[0][i][0:100:2,0:100:2,:])
        gt.append(data[1][i][0:100:2,0:100:2,:])
    XT = np.array(im).astype(np.single)
    YT = np.array(gt).astype(np.single)
    print('Training set')
    print(X.shape)
    print(Y.shape)
    print('Testing set')
    print(XT.shape)
    print(YT.shape)
    return X, Y, XT, YT

def sobel_output_shape(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 2)

def rand_image(model, X):
    YP = model(X)
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]: #vertical, dy, channel 0
            channel = 0
        if u[1] == v[1]: #horizontal, dy, channel 1
            channel = 1
        d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0
    L = ev.GetLabelsAtThreshold(G)
    img = np.zeros((SIZE[0], SIZE[1]), np.single)
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            img[i,j] = L[(i,j)]
    return img

def rand_error(YP, Y):              
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]: #vertical, dy, channel 0
            channel = 0
        if u[1] == v[1]: #horizontal, dy, channel 1
            channel = 1
        d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0
        nlabels_dict[u] = Y[0, u[0], u[1], 0]
        nlabels_dict[v] = Y[0, v[0], v[1], 0]    
    error = ev.FindRandErrorAtThreshold(G, nlabels_dict, 0.0)
    return error

def get_rand_weight(YP, Y):
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]: #vertical, dy, channel 0
            channel = 0
        if u[1] == v[1]: #horizontal, dy, channel 1
            channel = 1
        d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0
        nlabels_dict[u] = Y[0, u[0], u[1], 0]
        nlabels_dict[v] = Y[0, v[0], v[1], 0]

    [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
    posError = totalPos
    negError = 0.0
    WY = np.zeros((1, SIZE[0], SIZE[1], 1), np.single)
    SY = np.zeros((1, SIZE[0], SIZE[1], 1), np.single)
    for i in range(len(posCounts)):
        posError = posError - posCounts[i]
        negError = negError + negCounts[i]
        WS = posError - negError
        (u,v) = mstEdges[i]
        if u[0] == v[0]: #vertical, dy, channel 0
            channel = 0
        if u[1] == v[1]: #horizontal, dy, channel 1
            channel = 1

        WY[0, u[0], u[1], 0] += abs(WS)/2.0
        WY[0, v[0], v[1], 0] += abs(WS)/2.0
        if WS > 0.0:
            SY[0, u[0], u[1], 0] += 0.5                                  
            SY[0, v[0], v[1], 0] += 0.5  
        if WS < 0.0:
            SY[0, u[0], u[1], 0] += -0.5                                   
            SY[0, v[0], v[1], 0] += -0.5 
    # Std normalization
    totalW = np.sum(WY)
    if totalW != 0.0:
        WY = np.divide(WY, totalW)
    #SY = np.divide(SY, np.max(SY))
    return [WY, SY]

def get_maximin_weight(YP, Y):
    G = nx.grid_2d_graph(SIZE[0], SIZE[1])
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]: #vertical, dy, channel 0
            channel = 0
        if u[1] == v[1]: #horizontal, dy, channel 1
            channel = 1
        d['weight'] =  (YP[0, u[0], u[1], 0] + YP[0, v[0], v[1], 0])/2.0
        nlabels_dict[u] = Y[0, u[0], u[1], 0]
        nlabels_dict[v] = Y[0, v[0], v[1], 0]
    WY = np.zeros((1, SIZE[0], SIZE[1], 1), np.single)
    SY = np.zeros((1, SIZE[0], SIZE[1], 1), np.single)
    #build an MST
    mstEdges = ev.mstEdges(G)
    MST = nx.Graph()
    MST.add_edges_from(mstEdges)
    #get a random pair u,v
    u = (randint(0, SIZE[0]-1), randint(0, SIZE[1]-1))
    v = (randint(0, SIZE[0]-1), randint(0, SIZE[1]-1))
    while u == v:
        u = (randint(0, SIZE[0]-1), randint(0, SIZE[1]-1))
        v = (randint(0, SIZE[0]-1), randint(0, SIZE[1]-1))
    #find the maximin path between u and v on the MST
    path = nx.shortest_path(MST, source=u, target=v)

    #the maximin edge
    (us,vs) = min([edge for edge in nx.utils.pairwise(path)], key=lambda e: G.edges[e]['weight'])

    WY[0, us[0], us[1], 0] += 0.5  
    WY[0, vs[0], vs[1], 0] += 0.5  

    if Y[0, u[0], u[1], 0] == Y[0, v[0], v[1], 0]:
        SY[0, us[0], us[1], 0] += 0.5                               
        SY[0, vs[0], vs[1], 0] += 0.5  
    else:
        SY[0, us[0], us[1], 0] += -0.5                                  
        SY[0, vs[0], vs[1], 0] += -0.5  
    return [WY, SY]

def conv_block(input_tensor, num_filters):
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('relu')(encoder)
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    return decoder

def runExp(): 
    X, Y, XT, YT = getData()
    
    print('## Model ')
    input_image = tf.keras.layers.Input(shape=(50,50,3))
    encoder0_pool, encoder0 = encoder_block(input_image, 64)
    center = conv_block(encoder0_pool, 128)
    decoder0 = decoder_block(center, encoder0, 64)
    out = tf.keras.layers.Conv2D(1, (3,3), padding='same')(decoder0)
    model = tf.keras.Model(inputs=input_image, outputs=out)
    #model.summary()

    def myloss(model, X, Y):
        YP = model(X)
        WY, SY = get_rand_weight(YP, Y)
        loss = tf.losses.hinge_loss(labels=SY, logits=YP, weights=WY, reduction=tf.losses.Reduction.NONE)
        return loss

    def my_maximin_loss(model, X, Y):
        YP = model(X)
        WY, SY = get_maximin_weight(YP, Y)
        loss = tf.losses.hinge_loss(labels=SY, logits=YP, weights=WY, reduction=tf.losses.Reduction.NONE)
        return loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    loss_kruskal = []
    val_loss_kruskal = []
    rand_kruskal = []
    val_rand_kruskal =[]

    for epoch in range(NUM_EPOCHS):
        #WY, SY = get_rand_weight(model(X),Y)
        with tf.GradientTape() as tape:
            loss = myloss(model, X, Y)
           
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
        
        if (epoch+1) % 10 == 0:
            loss_kruskal.append(loss)

            v_loss = myloss(model, XT, YT)
            val_loss_kruskal.append(v_loss)

            pred_img = rand_image(model, XT)
            debugImages(YT[0,:,:,0], pred_img)

            err = rand_error(model(X), Y)
            v_err = rand_error(model(XT), YT)
            rand_kruskal.append(err)
            val_rand_kruskal.append(v_err)
            print("Epoch: {},         Loss: {}         Err: {}".format(epoch+1, loss, err))
            print("Epoch: {},     Val Loss: {}     Val Err: {}".format(epoch+1, v_loss, v_err))
    del model
    with open('train_kruskal.log', 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'loss_kruskal', 'val_loss_kruskal', 'rand_kruskal', 'val_rand_kruskal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(loss_kruskal)):
            writer.writerow({'epoch':(i+1)*10, 
                            'loss_kruskal':loss_kruskal[i].numpy(), 
                            'val_loss_kruskal':val_loss_kruskal[i].numpy(), 
                            'rand_kruskal':rand_kruskal[i], 
                            'val_rand_kruskal':val_rand_kruskal[i]})
    

####################################################################################################################################
####################################################################################################################################


if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    runExp()
