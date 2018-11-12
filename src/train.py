from data import *
import model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Init")
    model = model.unet()
    model_checkpoint = ModelCheckpoint('11-8-0.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    model_earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
    print('Getting data')
    
    file = open('data_image.p', 'rb')
    data = pickle.load(file)
    file.close()
    
    print('Done getting data')
    
    print('Start training')

    history = model.fit({'input_image': data[0], 'input_nlabels': data[1]}, data[2],
                    batch_size=2, epochs=100, verbose=1, callbacks=[model_checkpoint, model_earlyStopping], validation_split=0.2, shuffle=True)
    
    print('Done training')
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    print("Exit")