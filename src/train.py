from data import *
import model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

class BSDS500Sequence(Sequence):

    def __init__(self, images, nlabels, elabels, batch_size):
        self.images = images
        self.nlabels = nlabels
        self.elabels = elabels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_nlabels = self.nlabels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_elabels = self.elabels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_images), np.array(batch_nlabels), np.array(batch_elabels)

if __name__ == '__main__':
    print("Init")
    model = model.unet()
    model_checkpoint = ModelCheckpoint('C:/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    
    print('Processing data')
    file = open('save.p', 'rb')
    data = pickle.load(file)
    file.close()
    #val = ValData()
    print('Done processing data')
    
    print('Preparing dataset')
    dataset = BSDS500Sequence(data[0], data[1], data[2], 2)
    print('Done preparing dataset')
    
    print('Start training')
    history = model.fit({'images': data[0], 'nlabels': data[1]}, data[2],
                    batch_size=2, epochs=5, verbose=1, callbacks=[model_checkpoint], shuffle=True)
    #model.fit_generator(generator,steps_per_epoch=1000,epochs=5)
    print("Exit")