import os
import numpy as np
import tensorflow.keras.preprocessing.image as TFimage
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
TRAIN_GT_DIR = 'C:/data/BSDS500/data/groundTruth/train/'
TRAIN_IMG_DIR = 'C:/data/BSDS500/data/images/train/'
TEST_GT_DIR = 'C:/data/BSDS500/data/groundTruth/test/'
TEST_IMG_DIR = 'C:/data/BSDS500/data/images/test/'
VAL_GT_DIR = 'C:/data/BSDS500/data/groundTruth/val/'
VAL_IMG_DIR = 'C:/data/BSDS500/data/images/val/'

GT_EXT = '.mat'
IMG_EXT = '.jpg'
INPUT_SIZE = (481, 321, 3)
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
            image = TFimage.img_to_array(TFimage.load_img(path=TRAIN_IMG_DIR+ID+IMG_EXT, grayscale=False,
                                                target_size=None, interpolation='nearest'))
            groundTruth = loadmat(TRAIN_GT_DIR+ID+GT_EXT)
        elif p == 'val':
            image = TFimage.img_to_array(TFimage.load_img(path=VAL_IMG_DIR+ID+IMG_EXT, grayscale=False,
                                                target_size=None, interpolation='nearest'))
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

def TrainData():
    #images = np.array([Sample('train', str(ID))[0] for ID in TrainIDs()])
    #nlabels = np.array([Sample('train', str(ID))[1] for ID in TrainIDs()])
    #elabels = np.array([Sample('train', str(ID))[2] for ID in TrainIDs()])

    images = []
    nlabels = []
    elabels = []
    ids = TrainIDs()
    for i in range(1):
        s = Sample('train', str(ids[i]))
        images.append(s[0])
        nlabels.append(s[1])
        elabels.append(s[2])
    return np.array(images), np.array(nlabels), np.array(elabels)

def ValData():
    images = []
    nlabels = []
    elabels = []
    for ID in TrainIDs():
        s = Sample('val', str(ID))
        images.append(s[0])
        nlabels.append(s[1])
        elabels.append(s[2])
    return images, nlabels, elabels


def test_sample(images, nlabels, elabels):

    G = nx.grid_2d_graph(elabels.shape[0], elabels.shape[1])
    for (u,v,d) in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1

        d['weight'] = elabels[u[0], u[1], channel]

    theta = 0
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}

    result_image = np.zeros((elabels.shape[0], elabels.shape[1]))

    for j in range(result_image.shape[1]):
        for i in range(result_image.shape[0]):
            result_image[i,j] = L[(i,j)]

    fig=plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 4, 1)
    plt.imshow(images)
    fig.add_subplot(1, 4, 2)
    plt.imshow(np.squeeze(nlabels), cmap='nipy_spectral')
    fig.add_subplot(1, 4, 3)
    plt.imshow(result_image, cmap='nipy_spectral')
    #fig.add_subplot(1, 4, 4)
    #plt.imshow(elabels[:,:,1], cmap='nipy_spectral')

    plt.show()


if __name__ == '__main__':
    print("Init")
    ID = '2092'
    '''
    data = TrainData()
    f = open('data_image.p', 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    file = open('data_image.p', 'rb')
    data = pickle.load(file)
    file.close()
    '''
    data = TrainData()
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    test_sample(data[0][0], data[1][0], data[2][0])
    print("Exit")