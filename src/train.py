from data import *
import model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Init")
    
    print('Getting data')
    
    data = TrainData()
    
    print('Done getting data')
    
    model = model.unet()
    model_checkpoint = ModelCheckpoint('11-26.hdf5', monitor='loss',verbose=1, save_best_only=True)
    
    
    print('Start training')

    history = model.fit({'input_image': data[0], 'input_nlabels': data[1]}, data[2],
                    batch_size=1, epochs=100, verbose=1, callbacks=[model_checkpoint], shuffle=True)
    #model.save_weights('my_model_weights.h5')
    print('Done training')
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
    print("Exit")