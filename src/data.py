import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as TFimage
from scipy.io import loadmat
TRAIN_GT_DIR = '../data/BSDS500/data/groundTruth/train/'
TRAIN_IMG_DIR = '../data/BSDS500/data/images/train/'
TEST_GT_DIR = '../data/BSDS500/data/groundTruth/test/'
TEST_IMG_DIR = '../data/BSDS500/data/images/test/'
VAL_GT_DIR = '../data/BSDS500/data/groundTruth/val/'
VAL_IMG_DIR = '../data/BSDS500/data/images/val/'

GT_EXT = '.mat'
IMG_EXT = '.jpg'


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
            image = TFimage.img_to_array(TFimage.load_img(path=TRAIN_IMG_DIR+ID+IMG_EXT, grayscale=True,
                                                target_size=None, interpolation='nearest'))
            groundTruth = loadmat(TRAIN_GT_DIR+ID+GT_EXT)
        elif p == 'val':
            image = TFimage.img_to_array(TFimage.load_img(path=VAL_IMG_DIR+ID+IMG_EXT, grayscale=True,
                                                target_size=None, interpolation='nearest'))
            groundTruth = loadmat(VAL_GT_DIR+ID+GT_EXT)

        gtseg = groundTruth['groundTruth'][0,segId]['Segmentation'][0,0].astype(np.float32)
        if image.shape[0] < image.shape[1]:
            image = np.rot90(image, k=1, axes=(0,1))
            gtseg = np.rot90(gtseg, k=1, axes=(0,1))
        #3 channels: (intensity, horizontal grad, vertical grad)
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
        gtseg = np.delete(gtseg, 0, 0)
        gtseg = np.delete(gtseg, 0, 1)
        image = np.concatenate((image,
                        np.gradient(image, edge_order=2, axis=0),
                        np.gradient(image, edge_order=2, axis=1)),
                        axis=2)
        return(image, np.expand_dims(gtseg, axis=2))

def TrainData():
    images = np.array([Sample('train', str(ID))[0] for ID in TrainIDs()])
    gtseg = np.array([Sample('train', str(ID))[1] for ID in TrainIDs()])
    return images, gtseg

def ValData():
    images = np.array([Sample('val', str(ID))[0] for ID in ValIDs()])
    gtseg = np.array([Sample('val', str(ID))[1] for ID in ValIDs()])
    return images, gtseg

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

if __name__ == '__main__':
    print("Init")
    ID = '2092'
    TrainSample(ID)
    print("Exit")