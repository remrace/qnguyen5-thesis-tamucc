from data import *
import model
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import csv

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def write_time_log(time_callback, filename='default.log'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in enumerate(time_callback.times):
            writer.writerow({'epoch': k, 'time': v})

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
    #model_checkpoint_3 = ModelCheckpoint('C:/model/02-17-maximin-learning.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_rand_error', verbose=1, save_weights_only=True, period=2000)
    model_checkpoint_3_best_0 = ModelCheckpoint('C:/model/02-17-maximin-learning.best.0.hdf5', monitor='val_rand_error_0', verbose=1, save_best_only=True, save_weights_only=True)
    model_checkpoint_3_best_lowT = ModelCheckpoint('C:/model/02-17-maximin-learning.best.lowT.hdf5', monitor='val_rand_error_lowT', verbose=1, save_best_only=True, save_weights_only=True)
    csv_logger_3 = CSVLogger('C:/model/02-17-1-maximin-learning-training.log', append=True)
    history_3 = model_3.fit({'input_image': np.array(data[0][0:2]), 'input_nlabels': np.array(data[1][0:2])}, np.array(data[2][0:2]),
                            batch_size=1, epochs=1000, verbose=2, validation_split=0.5,
                            callbacks=[model_checkpoint_3_best_0, model_checkpoint_3_best_lowT, csv_logger_3, time_callback_3], shuffle=True)
    write_time_log(time_callback_3, 'C:/model/02-17-maximin-learning.log')
    print('Done training')
    #---------------------------------------------
    '''
    #---------------------------------------------
    print('Start training')
    model_0 = model.unet(mode='NO_CC_INFERENCE')
    time_callback_0 = TimeHistory()
    #model_checkpoint_0 = ModelCheckpoint('C:/model/02-17-no-inference.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_rand_error', verbose=1, save_weights_only=True, period=50)
    model_checkpoint_0_best_0 = ModelCheckpoint('C:/model/02-17-no-inference.best.0.hdf5', monitor='val_rand_error_0', verbose=1, save_best_only=True, save_weights_only=True)
    model_checkpoint_0_best_lowT = ModelCheckpoint('C:/model/02-17-no-inference.best.lowT.hdf5', monitor='val_rand_error_lowT', verbose=1, save_best_only=True, save_weights_only=True)
    csv_logger_0 = CSVLogger('C:/model/02-17-no-inference-training.log', append=False)
    history_0 = model_0.fit({'input_image': np.array(data[0][0:2]), 'input_nlabels': np.array(data[1][0:2])}, np.array(data[2][0:2]),
                            batch_size=1, epochs=200, verbose=2, validation_split=0.5,
                            callbacks=[model_checkpoint_0_best_0, model_checkpoint_0_best_lowT, csv_logger_0, time_callback_0], shuffle=True)
    write_time_log(time_callback_0, 'C:/model/02-17-no-inference.log')
    print('Done training')
    #---------------------------------------------
    #print('Start training')
    #model_1 = model.unet(mode='USE_CC_INFERENCE')
    #model_checkpoint_1 = ModelCheckpoint('C:/model/02-17-with-inference.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=False, period=50)
    #model_checkpoint_1_best = ModelCheckpoint('C:/model/02-17-with-inference.best.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=5)
    #csv_logger_1 = CSVLogger('C:/model/02-17-1-with-inference-training.log', append=True)
    #time_callback = TimeHistory()
    
    #model_earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=100, verbose=1, mode='auto')
    
    
    #history_1 = model_1.fit({'input_image': np.array(data[0]), 'input_nlabels': np.array(data[1])}, np.array(data[2]),
    #                        batch_size=1, epochs=1000, verbose=1, validation_split=0.2,
    #                        callbacks=[model_checkpoint_1, model_checkpoint_1_best, csv_logger_1, time_callback], shuffle=True)

    #print('Done training')
   

    
    
    

    '''
    with open('C:/model/01-21-history.hdf5', 'wb') as file_pi:
        pickle.dump([history_0.history, history_1.history], file_pi)
    
    with open('C:/model/01-21-history.hdf5', 'rb') as file_pi:
        history = pickle.load(file_pi)

    print(history[0]['loss'], history[1]['val_loss'])

    fig=plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 2, 1)
    plt.plot(history[0]['loss'])
    plt.plot(history[0]['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc='upper left')
    
    fig.add_subplot(1, 2, 2)
    plt.plot(history[1]['loss'])
    plt.plot(history[1]['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc='upper left')
    


    plt.show()
    '''
    print("Exit")