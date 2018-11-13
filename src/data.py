import os
import numpy as np
import tensorflow.keras.preprocessing.image as TFimage
import networkx as nx
import pickle
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
NUM_OUTPUTS = 1
KERNEL_SIZE = 3
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
        
        
        '''
        #delete one row and one column so the data fits the unet
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
        gtseg = np.delete(gtseg, 0, 0)
        gtseg = np.delete(gtseg, 0, 1)
        
        
        image = np.concatenate((image,
                        np.gradient(image, edge_order=2, axis=0),
                        np.gradient(image, edge_order=2, axis=1)),
                        axis=2)
        '''
        nlabels = np.expand_dims(gtseg, axis=2)
        #elabels in a 2d array, padded with -1
        '''
        elabels = np.negative(np.ones((480, 320, 2), np.float32))
        G = nx.grid_2d_graph(gtseg.shape[0], gtseg.shape[1])
        for (u,v,d) in G.edges(data = True):
            

            if u[0] == v[0]:
                channel = 0
            else:
                channel = 1

            if abs(labelu - labelv) < 1.0:
                elabels[u[0], u[1], channel] = 1.0
            else:
                elabels[u[0], u[1], channel] = -1.0
        '''
        #try elabels as vector
        X_train = np.zeros( (N, D), np.float32)
        elabels = np.zeros( (N, 1), np.float32)
        upto = 0

        G = nx.grid_2d_graph(gtseg.shape[0], gtseg.shape[1])
        for (u,v) in G.edges():
            
            uimg = image[u[0]:(u[0]+KERNEL_SIZE), u[1]:(u[1]+KERNEL_SIZE), :]
            vimg = image[v[0]:(v[0]+KERNEL_SIZE), v[1]:(v[1]+KERNEL_SIZE), :]
            
            if uimg.shape != (KERNEL_SIZE,KERNEL_SIZE,3):
                uimg = np.pad(uimg, ((0,KERNEL_SIZE-uimg.shape[0]), (0, KERNEL_SIZE-uimg.shape[1]), (0,0)), mode='edge')
            if vimg.shape != (KERNEL_SIZE,KERNEL_SIZE,3):
                vimg = np.pad(vimg, ((0,KERNEL_SIZE-vimg.shape[0]), (0, KERNEL_SIZE-vimg.shape[1]), (0,0)), mode='edge')
            
            aimg = (uimg + vimg) / 2.0
            X_train[upto,:] = np.reshape(aimg, (1, D))
            
            labelu = gtseg[u[0], u[1]]
            labelv = gtseg[v[0], v[1]]
            if abs(labelu - labelv) < 1.0:
                elabels[upto, 0] = 1.0
            else:
                elabels[upto, 0] = -1.0
            upto = upto + 1

        return (X_train, nlabels, elabels)

def TrainData():
    #images = np.array([Sample('train', str(ID))[0] for ID in TrainIDs()])
    #nlabels = np.array([Sample('train', str(ID))[1] for ID in TrainIDs()])
    #elabels = np.array([Sample('train', str(ID))[2] for ID in TrainIDs()])

    images = []
    nlabels = []
    elabels = []
    ids = TrainIDs()
    for i in range(5):
        s = Sample('train', str(ids[i]))
        images.append(s[0])
        nlabels.append(s[1])
        elabels.append(s[2])
    return np.array(images, np.float32), np.array(nlabels, np.float32), np.array(elabels, np.float32)

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


'''
def TrainGenerator():
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    images, gtsegs = TrainData()
    image_datagen = TFimage.ImageDataGenerator(**data_gen_args)
    gtseg_datagen = TFimage.ImageDataGenerator(**data_gen_args)
    seed = 1
    image_datagen.fit(images, augment=False, seed=seed)
    gtseg_datagen.fit(gtsegs, augment=False, seed=seed)
    image_generator = image_datagen.flow(images, batch_size=2, shuffle = True, seed = seed)
    gtseg_generator = gtseg_datagen.flow(gtsegs, batch_size=2, shuffle = True, seed = seed)
    train_generator = zip(image_generator, gtseg_generator)
    return train_generator

def ValGenerator():
    images, gtsegs = ValData()
    val_datagen = TFimage.ImageDataGenerator()
    val_generator = val_datagen.flow(images, batch_size=1)
    return val_generator
'''
if __name__ == '__main__':
    print("Init")
    ID = '2092'
    
    data = TrainData()
    f = open('data_image.p', 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    '''
    file = open('data_vect.p', 'rb')
    data = pickle.load(file)
    file.close()
    '''
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    
    print("Exit")