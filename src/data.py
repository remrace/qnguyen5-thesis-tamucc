import os
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import data, io, filters, util
from sklearn.preprocessing import scale
import sklearn.feature_extraction
import random
import SegEval as ev
TRAIN_GT_DIR = 'C:/data/BSDS500/data/groundTruth/train/'
TRAIN_IMG_DIR = 'C:/data/BSDS500/data/images/train/'
TEST_GT_DIR = 'C:/data/BSDS500/data/groundTruth/test/'
TEST_IMG_DIR = 'C:/data/BSDS500/data/images/test/'
VAL_GT_DIR = 'C:/data/BSDS500/data/groundTruth/val/'
VAL_IMG_DIR = 'C:/data/BSDS500/data/images/val/'

GT_EXT = '.mat'
IMG_EXT = '.jpg'
INPUT_SIZE = (128, 128, 1)
KERNEL_SIZE = 5
N = (INPUT_SIZE[0]-1) * INPUT_SIZE[1] + (INPUT_SIZE[1]-1) * INPUT_SIZE[0]
D = KERNEL_SIZE * KERNEL_SIZE * 3  


def TrainIDs(n=None):
    return sorted([int(os.path.splitext(filename)[0]) for filename in os.listdir(TRAIN_GT_DIR)])
        
def TestIDs(n=None):
    return sorted([int(os.path.splitext(filename)[0]) for filename in os.listdir(TEST_GT_DIR)])

def ValIDs(n=None):
    return sorted([int(os.path.splitext(filename)[0]) for filename in os.listdir(VAL_GT_DIR)])


def Sample(p = 'train', ID = None):
    if ID == None:
        print('TrainSample needs to know the sample id.')
        return
    else:
        segId = 0
        if p == 'train':
            image = io.imread(TRAIN_IMG_DIR+ID+IMG_EXT)
            groundTruth = loadmat(TRAIN_GT_DIR+ID+GT_EXT)
        elif p == 'val':
            image = io.imread(TRAIN_IMG_DIR+ID+IMG_EXT)
            groundTruth = loadmat(VAL_GT_DIR+ID+GT_EXT)

        gtseg = groundTruth['groundTruth'][0,segId]['Segmentation'][0,0].astype(np.float32)
        if image.shape[0] < image.shape[1]:
            image = np.rot90(image, k=1, axes=(0,1))
            gtseg = np.rot90(gtseg, k=1, axes=(0,1))
        
        
        #delete one row and one column so the data fits the unet
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
        gtseg = np.delete(gtseg, 0, 0)
        gtseg = np.delete(gtseg, 0, 1)
        
        
        elabels = np.ones((gtseg.shape[0], gtseg.shape[1], 2), np.float32)

        G = nx.grid_2d_graph(gtseg.shape[0], gtseg.shape[1])
        for (u,v,d) in G.edges(data = True):
            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1

            if abs(gtseg[u] - gtseg[v]) > 0.0:
                elabels[u[0], u[1], channel] = -1.0


        nlabels = np.expand_dims(gtseg, axis=2)
        return (image, nlabels, elabels)

def TrainData(num_samples=20):
    #images = np.array([Sample('train', str(ID))[0] for ID in TrainIDs()])
    #nlabels = np.array([Sample('train', str(ID))[1] for ID in TrainIDs()])
    #elabels = np.array([Sample('train', str(ID))[2] for ID in TrainIDs()])

    images = []
    nlabels = []
    elabels = []
    ids = TrainIDs()
    for i in range(num_samples):
        s = Sample('train', str(ids[i]))
        images.append(s[0])
        nlabels.append(s[1])
        elabels.append(s[2])
    return np.array(images), np.array(nlabels), np.array(elabels)

def DataPatches(p = 'train', ID = None):
    if ID == None:
        print('needs to know the sample id.')
        return
    segId = 0
    if p == 'train':
        image = io.imread(TRAIN_IMG_DIR+ID+IMG_EXT, as_grey=True)
        groundTruth = loadmat(TRAIN_GT_DIR+ID+GT_EXT)
    elif p == 'val':
        image = io.imread(TRAIN_IMG_DIR+ID+IMG_EXT, as_grey=True)
        groundTruth = loadmat(VAL_GT_DIR+ID+GT_EXT)
    image = scale(image)
    gtseg = groundTruth['groundTruth'][0,segId]['Segmentation'][0,0].astype(np.float32)
    image_patches = sklearn.feature_extraction.image.extract_patches_2d(image, (INPUT_SIZE[0], INPUT_SIZE[1]), max_patches=100, random_state=3)
    gtseg_pathces = sklearn.feature_extraction.image.extract_patches_2d(gtseg, (INPUT_SIZE[0], INPUT_SIZE[1]), max_patches=100, random_state=3)
    image_patches = np.expand_dims(image_patches, axis=-1)

    n, y, x = gtseg_pathces.shape
    crop = 18
    gtseg_pathces = gtseg_pathces[:,crop:y-crop,crop:x-crop]
    elabels = np.ones((gtseg_pathces.shape[0], gtseg_pathces.shape[1], gtseg_pathces.shape[2], 2), np.float32)
    for i in range(len(gtseg_pathces)):
        print(i)
        seg = gtseg_pathces[i]
        G = nx.grid_2d_graph(seg.shape[0], seg.shape[1])
        for (u,v,d) in G.edges(data = True):
            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1
            if abs(seg[u] - seg[v]) > 0.0:
                elabels[i, u[0], u[1], channel] = -1.0
    nlabels = np.expand_dims(gtseg_pathces, axis=-1)
    return image_patches, nlabels, elabels


def ValData(num_samples=20):
    #images = np.array([Sample('train', str(ID))[0] for ID in TrainIDs()])
    #nlabels = np.array([Sample('train', str(ID))[1] for ID in TrainIDs()])
    #elabels = np.array([Sample('train', str(ID))[2] for ID in TrainIDs()])

    images = []
    nlabels = []
    elabels = []
    ids = ValIDs()
    random_ids = random.sample(ids, num_samples)
    for id in random_ids:
        s = Sample('val', str(id))
        images.append(s[0])
        nlabels.append(s[1])
        elabels.append(s[2])
    return np.array(images), np.array(nlabels), np.array(elabels)


def test_sample(images, nlabels, elabels):
    G = nx.grid_2d_graph(elabels.shape[0], elabels.shape[1])
    for (u,v,d) in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1

        d['weight'] = elabels[u[0], u[1], channel]

    lowT, lowE = ev.FindMinEnergyThreshold(G)
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=lowT])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}

    result_image = np.zeros((elabels.shape[0], elabels.shape[1]))

    for j in range(result_image.shape[1]):
        for i in range(result_image.shape[0]):
            result_image[i,j] = L[(i,j)]

    fig=plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 4, 1)
    plt.imshow(np.squeeze(images))
    fig.add_subplot(1, 4, 2)
    plt.imshow(np.squeeze(nlabels), cmap='nipy_spectral')
    fig.add_subplot(1, 4, 3)
    plt.imshow(result_image, cmap='nipy_spectral')
    #fig.add_subplot(1, 4, 4)
    #plt.imshow(elabels[:,:,1], cmap='nipy_spectral')

    plt.show()


if __name__ == '__main__':
    print("Init")
    
    trainData = DataPatches(ID='94079')
    testData = DataPatches(ID='100075')
    
    f = open('train_data_patches_scaled_bears.p', 'wb')
    pickle.dump(trainData, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    f = open('test_data_patches_scaled_bears.p', 'wb')
    pickle.dump(testData, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    file = open('train_data_patches_scaled_bears.p', 'rb')
    readdata = pickle.load(file)
    file.close()
    print(readdata[0].shape, readdata[1].shape, readdata[2].shape)
    test_sample(readdata[0][5], readdata[1][5], readdata[2][5])

    file = open('test_data_patches_scaled_bears.p', 'rb')
    readdata = pickle.load(file)
    file.close()
    print(readdata[0].shape, readdata[1].shape, readdata[2].shape)
    test_sample(readdata[0][5], readdata[1][5], readdata[2][5])

    print("Exit")