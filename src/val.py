from data import *
import random
import model
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import SegEval as ev
import SegGraph as seglib
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy import stats
INPUT_SIZE = (100, 100, 3)
NUM_EDGES = INPUT_SIZE[0]*(INPUT_SIZE[1]-1) + (INPUT_SIZE[0]-1)*INPUT_SIZE[1]
def test(USE_CC_INFERENCE = False):
    my_model = model.unet(USE_CC_INFERENCE=False, pretrained_weights='C:/model/02-09-no-inference.hdf5')
   
    print('Getting data')
    f = open('../synimage/test.p', 'rb')
    data = pickle.load(f)
    f.close()
    print('Done getting data')
    
    x = np.array(data[0])
    y = np.array(data[1])
    z = np.array(data[2])
    
    pred = my_model.predict([x, y], batch_size=1)

    result = []

    for idx in range(x.shape[0]):
        images, nlabels, elabels = x[idx], y[idx], z[idx]
        G = nx.grid_2d_graph(elabels.shape[0], elabels.shape[1])
        gtLabels = dict()
        for (u,v,d) in G.edges(data = True):
            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1
            gtLabels[u] = nlabels[u[0], u[1], 0]
            gtLabels[v] = nlabels[v[0], v[1], 0]
            d['weight'] = pred[idx, u[0], u[1], channel]
        
        lowT, lowE, posCountsRand, negCountsRand, mstEdges, mstEdgeWeights, totalPosRand, totalNegRand = ev.FindMinEnergyAndRandCounts(G, gtLabels)
        randE = ev.FindRandErrorAtThreshold(G, gtLabels, lowT)
        lowT_R, lowE_R = ev.FindBestRandThreshold(posCountsRand, negCountsRand, mstEdges, mstEdgeWeights)
        print(lowT, lowE, randE)
        print(lowT_R, lowE_R)

        L = seglib.GetLabelsAtThreshold(G, lowT)

        result_image = np.zeros((elabels.shape[0], elabels.shape[1]), np.float32)

        for j in range(result_image.shape[1]):
            for i in range(result_image.shape[0]):
                result_image[i,j] = L[(i,j)]

        fig=plt.figure()
        fig.add_subplot(1, 5, 1)
        plt.imshow(np.squeeze(images))
        fig.add_subplot(1, 5, 2)
        plt.imshow(np.squeeze(nlabels))
        fig.add_subplot(1, 5, 3)
        plt.imshow(pred[idx, :,:,0])
        fig.add_subplot(1, 5, 4)
        plt.imshow(pred[idx, :,:,1])
        fig.add_subplot(1, 5, 5)
        plt.imshow(result_image)
        plt.title('lowT: ' + str(lowT) + 'lowE: ' + str(lowE))

        result.append([lowT, lowT_R, randE, lowE_R])

        if USE_CC_INFERENCE:
            fname = 'test/with-inference-' + str(idx) + '.png'
        else:
            fname = 'test/no-inference-' + str(idx) + '.png'
        fig.savefig(fname)
    f = open('result.p', 'wb')
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == '__main__':
    print('Init')
    test(USE_CC_INFERENCE = False)
    #test(num_of_samples = 50, USE_CC_INFERENCE = True, seed=3)
    print('Exit')
