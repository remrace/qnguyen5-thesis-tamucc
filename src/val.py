from data import *
import random
import model
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import SegEval as ev
import SegGraph as seglib
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy import stats
INPUT_SIZE = (128-36, 128-36, 1)
NUM_EDGES = INPUT_SIZE[0]*(INPUT_SIZE[1]-1) + (INPUT_SIZE[0]-1)*INPUT_SIZE[1]
def test(num_of_samples = 5, USE_CC_INFERENCE = False, seed=None):
    if USE_CC_INFERENCE:
        my_model = model.unet(USE_CC_INFERENCE, pretrained_weights='C:/model/01-16-with-inference.hdf5')
    else:
        my_model = model.unet(USE_CC_INFERENCE, pretrained_weights='C:/model/01-16-no-inference.hdf5')
   
    print('Getting data')
    file = open('test_data_patches_scaled.p', 'rb')
    data = pickle.load(file)
    file.close()
    print('Done getting data')
    
    if seed is not None:
        random.seed(seed)
    
    x = []
    y = []
    z = []

    indices = random.sample(range(len(data[0])),num_of_samples)
    for idx in indices:
        x.append(data[0][idx])
        y.append(data[1][idx])
        z.append(data[2][idx])
    
    pred = my_model.predict([x, y], batch_size=1)
    print(pred[0].shape)

    for idx in range(num_of_samples):
        images, nlabels, elabels = x[idx], y[idx], z[idx]
        print('Got data')
        G = nx.grid_2d_graph(elabels.shape[0], elabels.shape[1])
        for (u,v,d) in G.edges(data = True):
            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1

            d['weight'] = pred[idx, u[0], u[1], channel]
        print('Got graph')
        
        lowT, lowE = ev.FindMinEnergyThreshold(G)
        print(lowT, lowE)
        L = seglib.GetLabelsAtThreshold(G, lowT)

        result_image = np.zeros((elabels.shape[0], elabels.shape[1]), np.float32)

        for j in range(result_image.shape[1]):
            for i in range(result_image.shape[0]):
                result_image[i,j] = L[(i,j)]

        fig=plt.figure(figsize=(8, 4))
        fig.add_subplot(1, 5, 1)
        plt.imshow(np.squeeze(images))
        fig.add_subplot(1, 5, 2)
        plt.imshow(np.squeeze(nlabels), cmap='nipy_spectral')
        fig.add_subplot(1, 5, 3)
        plt.imshow(pred[0, :,:,0], cmap='nipy_spectral')
        fig.add_subplot(1, 5, 4)
        plt.imshow(pred[0, :,:,1], cmap='nipy_spectral')
        fig.add_subplot(1, 5, 5)
        plt.imshow(result_image, cmap='nipy_spectral')
        plt.title('lowT: ' + str(lowT) + 'lowE: ' + str(lowE))
        if USE_CC_INFERENCE:
            fname = 'with-inference-' + str(indices[idx]) + '.png'
        else:
            fname = 'no-inference-' + str(indices[idx]) + '.png'
        fig.savefig(fname)
    

if __name__ == '__main__':
    print('Init')
    test(num_of_samples = 5, USE_CC_INFERENCE = False, seed=3)
    #test(num_of_samples = 5, USE_CC_INFERENCE = True, seed=3)
    print('Exit')
