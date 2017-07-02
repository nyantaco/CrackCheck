# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img

nb_epoch = 5

temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    
def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

def draw_images(temp_dir):
    images = glob.glob(os.path.join(temp_dir, "*.jpg"))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(9):
        img = load_img(images[i])
        plt.subplot(gs[i])
        plt.imshow(img, aspect='auto')
        plt.axis("off")
    plt.savefig("Generated Images")


if __name__ == '__main__':
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    

    
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        'images',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
#        ,
#        save_to_dir=temp_dir, 
#        save_prefix='img',
#        save_format='jpg')
    
    validation_generator = test_datagen.flow_from_directory(
        'images',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    
#    i = 0
#    for batch in train_generator:
#        i += 1
#        if i > 2:
#            break
#    draw_images(temp_dir)

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=200,
        nb_epoch=nb_epoch
        ,
        validation_data=validation_generator,
        nb_val_samples=50)
    
    model.save_weights(os.path.join(result_dir, 'cnn.h5'))
    save_history(history, os.path.join(result_dir, 'cnn.txt'))
    
    