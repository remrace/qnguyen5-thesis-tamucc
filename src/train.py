from data import *
import model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Init")
    
    print('Getting data')
    f = open('../synimage/train.p', 'rb')
    data = pickle.load(f)
    f.close()
    print('Done getting data')
    
    #model_0 = model.unet(USE_CC_INFERENCE=False)
    #model_checkpoint_0 = ModelCheckpoint('C:/model/02-09-no-inference.hdf5', monitor='val_loss',verbose=1, save_weights_only=False, period=50)
    #csv_logger_0 = CSVLogger('C:/model/02-09-no-inference-training.log')

    model_1 = model.unet(USE_CC_INFERENCE=True, pretrained_weights='C:/model/02-09-no-inference.hdf5')
    model_checkpoint_1 = ModelCheckpoint('C:/model/02-09-with-inference.hdf5', monitor='val_loss',verbose=1, save_weights_only=False, period=50)
    csv_logger_1 = CSVLogger('C:/model/02-09-with-inference-training.log')
    #model_earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=100, verbose=1, mode='auto')
    
    print('Start training')
    #print('USE_CC_INFERENCE: NO')
    #history_0 = model_0.fit({'input_image': np.array(data[0]), 'input_nlabels': np.array(data[1])}, np.array(data[2]),
    #                batch_size=1, epochs=200, verbose=1, validation_split=0.2, callbacks=[model_checkpoint_0, csv_logger_0], shuffle=True)
    print('USE_CC_INFERENCE: YES')
    history_1 = model_1.fit({'input_image': np.array(data[0]), 'input_nlabels': np.array(data[1])}, np.array(data[2]),
                    batch_size=1, epochs=200, verbose=1, validation_split=0.2, callbacks=[model_checkpoint_1, csv_logger_1], shuffle=True)
    
    print('Done training')

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