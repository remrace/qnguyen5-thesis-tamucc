from data import *
import model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Init")
    
    print('Getting data')
    
    file = open('gray-image_nlabels_elabels.p', 'rb')
    data = pickle.load(file)
    file.close()
    print('Done getting data')
    
    model = model.unet(USE_CC_INFERENCE=False)
    model_checkpoint = ModelCheckpoint('C:/model/12-2-2018-3x3.hdf5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True)
    #model_earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5, verbose=1, mode='auto')
    
    print('Start training')

    history = model.fit({'input_image': data[0][0:50], 'input_nlabels': data[1][0:50]}, data[2][0:50],
                    batch_size=1, epochs=100, verbose=1, validation_split=0.2, callbacks=[model_checkpoint], shuffle=True)
    #model.save_weights('my_model_weights.h5')
    print('Done training')
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
    print("Exit")