import numpy as np
import random as rn
np.random.seed(37)
rn.seed(142857)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["PYTHONHASHSEED"]="0"
import tensorflow as tf
tf.set_random_seed(1357)
from data import *
import model
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import csv

ROOT = 'C:/model/'
DATE = '02-19'
SIZE = (64, 64)

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def write_time_log(time_callback, filename='default.log'):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in enumerate(time_callback.times):
            writer.writerow({'epoch': k, 'time': v})

def train(mode, BATCH_SIZE=1, EPOCHS=200, VALIDATION_SPLIT=0.5):
    print('Start training')
    model_0 = model.unet(mode)
    time_callback_0 = TimeHistory()
    #model_checkpoint_0 = ModelCheckpoint(ROOT + DATE + str(SIZE) + '-no-inference.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_rand_error', verbose=1, save_weights_only=True, period=50)
    model_checkpoint_0_best_0 = ModelCheckpoint(ROOT + DATE + str(SIZE) + str(mode) + '.best.0.hdf5', monitor='val_rand_error_0', verbose=1, save_best_only=True, save_weights_only=True)
    model_checkpoint_0_best_lowT = ModelCheckpoint(ROOT + DATE + str(SIZE) + str(mode) + '.best.lowT.hdf5', monitor='val_rand_error_lowT', verbose=1, save_best_only=True, save_weights_only=True)
    csv_logger_0 = CSVLogger(ROOT + DATE + str(SIZE) + str(mode) + '-training.log', append=True)
    history_0 = model_0.fit({'input_image': np.array(data[0][0:2])[:,0:SIZE[0],0:SIZE[1],:], 'input_nlabels': np.array(data[1][0:2])[:,0:SIZE[0],0:SIZE[1],:]}, np.array(data[2][0:2])[:,0:SIZE[0],0:SIZE[1],:], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_split=VALIDATION_SPLIT, callbacks=[model_checkpoint_0_best_0, model_checkpoint_0_best_lowT, csv_logger_0, time_callback_0], shuffle=True)
    write_time_log(time_callback_0, ROOT + DATE + str(SIZE) + str(mode) +'-time.log')
    print('Done training')
    return history_0


if __name__ == '__main__':
    print("Init")
    
    print('Getting data')
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()
    print('Done getting data')
    
    '''
    #---------------------------------------------
    print('Start training')
    model_3 = model.unet(mode='MAXIMIN_LEARNING')
    time_callback_3 = TimeHistory()
    #model_checkpoint_3 = ModelCheckpoint(ROOT + DATE + str(SIZE) + '-maximin-learning.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_rand_error', verbose=1, save_weights_only=True, period=2000)
    model_checkpoint_3_best_0 = ModelCheckpoint(ROOT + DATE + str(SIZE) + '-maximin-learning.best.0.hdf5', monitor='val_rand_error_0', verbose=1, save_best_only=True, save_weights_only=True)
    model_checkpoint_3_best_lowT = ModelCheckpoint(ROOT + DATE + str(SIZE) + '-maximin-learning.best.lowT.hdf5', monitor='val_rand_error_lowT', verbose=1, save_best_only=True, save_weights_only=True)
    csv_logger_3 = CSVLogger(ROOT + DATE + str(SIZE) + '-maximin-learning-training.log', append=True)
    history_3 = model_3.fit({'input_image': np.array(data[0][0:2])[:,0:SIZE[0],0:SIZE[1],:], 'input_nlabels': np.array(data[1][0:2])[:,0:SIZE[0],0:SIZE[1],:]}, np.array(data[2][0:2])[:,0:SIZE[0],0:SIZE[1],:],
                            batch_size=1, epochs=100, verbose=2, validation_split=0.5,
                            callbacks=[model_checkpoint_3_best_0, model_checkpoint_3_best_lowT, csv_logger_3, time_callback_3], shuffle=True)
    write_time_log(time_callback_3, ROOT + DATE + str(SIZE) + '-maximin-learning.log')
    print('Done training')
    #---------------------------------------------
    '''
    #---------------------------------------------
    for i in range(2):
        train(mode='NO_CC_INFERENCE', BATCH_SIZE=1, EPOCHS=100,VALIDATION_SPLIT=0.5)
        train(mode='MAXIMIN_LEARNING', BATCH_SIZE=1, EPOCHS=100,VALIDATION_SPLIT=0.5)
    #---------------------------------------------
    #print('Start training')
    #model_1 = model.unet(mode='USE_CC_INFERENCE')
    #model_checkpoint_1 = ModelCheckpoint(ROOT + DATE + str(SIZE) + '-with-inference.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=False, period=50)
    #model_checkpoint_1_best = ModelCheckpoint(ROOT + DATE + str(SIZE) + '-with-inference.best.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=5)
    #csv_logger_1 = CSVLogger(ROOT + DATE + str(SIZE) + '-with-inference-training.log', append=True)
    #time_callback = TimeHistory()
    
    #model_earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=100, verbose=1, mode='auto')
    
    
    #history_1 = model_1.fit({'input_image': np.array(data[0]), 'input_nlabels': np.array(data[1])}, np.array(data[2]),
    #                        batch_size=1, epochs=1000, verbose=1, validation_split=0.2,
    #                        callbacks=[model_checkpoint_1, model_checkpoint_1_best, csv_logger_1, time_callback], shuffle=True)

    #print('Done training')
    print("Exit")