import data
import model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler



if __name__ == '__main__':
    print("Init")
    generator = data.TrainGenerator()
    model = model.unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    print('Start training')
    model.fit_generator(generator,steps_per_epoch=100,epochs=3,callbacks=[model_checkpoint])
    #model.fit_generator(generator,steps_per_epoch=1000,epochs=5)
    print("Exit")