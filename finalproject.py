# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:54:59 2018

@author: dylan
"""




import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
import timeit


X=[]
Z=[]
IMG_SIZE=200
DAISY='./flowers/daisy'
SUNFLOWER='./flowers/sunflower'
TULIP='./flowers/tulip'
DANDELION='./flowers/dandelion'
ROSE='./flowers/rose'

def assign_label(img,flower_type):
    return flower_type

def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        try:
            path = os.path.join(DIR,img)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
            X.append(np.array(img))
            Z.append(str(label))
        except Exception as e:
            print(str(e))
        
        
make_train_data('Daisy',DAISY)
print(len(X))
make_train_data('Sunflower',SUNFLOWER)
print(len(X))
make_train_data('Tulip',TULIP)
print(len(X))
make_train_data('Dandelion',DANDELION)
print(len(X))
make_train_data('Rose',ROSE)
print(len(X))


        

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=20)

np.random.seed(20)
rn.seed(20)
tf.set_random_seed(20)

model2 = Sequential()
model2.add(Conv2D(filters = 64, kernel_size = (11,11),padding = 'valid', strides=(4,4), activation ='relu', input_shape = (200,200,3)))
model2.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2.add(BatchNormalization())



model2.add(Conv2D(filters = 96, kernel_size = (5,5),padding = 'same', activation ='relu'))
model2.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2.add(BatchNormalization())


model2.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', activation = 'relu'))
model2.add(Conv2D(filters =256, kernel_size = (3,3),padding = 'same', activation = 'relu'))
model2.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2.add(Dropout(0.5))

model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(5, activation = "softmax"))

batch_size=128
epochs=100

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model2.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model2.summary()

start2 = timeit.default_timer()

History = model2.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
stop2 = timeit.default_timer()


plt.savefig('examplepictures.png')
plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model2loss.png')
plt.clf()


plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model2accuracy.png')
plt.clf()

model2b = Sequential()
model2b.add(Conv2D(filters = 32, kernel_size = (11,11),padding = 'valid', strides=(4,4), activation ='relu', input_shape = (200,200,3)))
model2b.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2b.add(BatchNormalization())



model2b.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'same', activation ='relu'))
model2b.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2b.add(BatchNormalization())


model2b.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same', activation = 'relu'))
model2b.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', activation = 'relu'))
model2b.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2b.add(Dropout(0.5))

model2b.add(Flatten())
model2b.add(Dense(512, activation='relu'))
model2b.add(Dropout(0.5))
model2b.add(Dense(5, activation = "softmax"))

batch_size=128
epochs=100

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model2b.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model2b.summary()

start2b = timeit.default_timer()

History = model2b.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

stop2b = timeit.default_timer()

plt.savefig('examplepictures.png')
plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model2bloss.png')
plt.clf()


plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model2baccuracy.png')
plt.clf()

model2c = Sequential()
model2c.add(Conv2D(filters = 64, kernel_size = (11,11),padding = 'valid', strides=(4,4), activation ='relu', input_shape = (200,200,3)))
model2c.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2c.add(BatchNormalization())



model2c.add(Conv2D(filters = 96, kernel_size = (5,5),padding = 'same', activation ='relu'))
model2c.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model2c.add(BatchNormalization())


model2c.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', activation = 'relu'))
model2c.add(Conv2D(filters =256, kernel_size = (3,3),padding = 'same', activation = 'relu'))
model2c.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))


model2c.add(Flatten())
model2c.add(Dense(512, activation='relu'))
model2c.add(Dropout(0.5))
model2c.add(Dense(512, activation='relu'))
model2c.add(Dropout(0.5))
model2c.add(Dense(5, activation = "softmax"))

batch_size=128
epochs=100

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model2c.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model2c.summary()

start2c = timeit.default_timer()

History = model2c.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

stop2c = timeit.default_timer()

plt.savefig('examplepictures.png')
plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model2closs.png')
plt.clf()


plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model2caccuracy.png')
plt.clf()



model1 = Sequential()
model1.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same',activation ='relu', input_shape = (200,200,3)))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model1.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model1.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model1.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(5, activation = "softmax"))



batch_size=128
epochs= 100
from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model1.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model1.summary()

start1 = timeit.default_timer()

History = model1.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

stop1 = timeit.default_timer()

plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model1loss.png')
plt.clf()

plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model1accuracy.png')
plt.clf()


model1b = Sequential()
model1b.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same',activation ='relu', input_shape = (200,200,3)))
model1b.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model1b.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
 

model1b.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))

model1b.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))


model1b.add(Flatten())
model1b.add(Dense(512, activation='relu'))
model1b.add(Dropout(0.5))
model1b.add(Dense(5, activation = "softmax"))



batch_size=128
epochs= 100
from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model1b.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model1b.summary()

start1b = timeit.default_timer()

History = model1b.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

stop1b= timeit.default_timer()

plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model1bloss.png')
plt.clf()

plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model1baccuracy.png')
plt.clf()

model1c = Sequential()
model1c.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same',activation ='relu', input_shape = (200,200,3)))
model1c.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model1c.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model1c.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model1c.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model1c.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model1c.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same',activation ='relu'))
model1c.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model1c.add(Flatten())
model1c.add(Dense(512, activation='relu'))
model1c.add(Dropout(0.5))
model1c.add(Dense(5, activation = "softmax"))



batch_size=128
epochs= 100
from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model1c.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model1c.summary()

start1c = timeit.default_timer()

History = model1c.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

stop1c = timeit.default_timer()

plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model1closs.png')
plt.clf()

plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model1caccuracy.png')
plt.clf()



model3 = Sequential()
model3.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (200,200,3)))
model3.add(MaxPooling2D(pool_size=(2,2)))


model3.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
 

model3.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model3.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model3.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model3.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(5, activation = "softmax"))

batch_size=128
epochs= 100
from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


datagen.fit(x_train)

model3.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model3.summary()

start3 = timeit.default_timer()

History = model3.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

stop3 = timeit.default_timer()

plt.clf()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model3loss.png')
plt.clf()

plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.tight_layout(pad=2)
plt.savefig('model3accuracy.png')
plt.clf()



score2a = model2.evaluate(x_test, y_test, verbose=0)
score2b = model2b.evaluate(x_test, y_test, verbose=0)
score2c = model2c.evaluate(x_test, y_test, verbose=0)
score1a = model1.evaluate(x_test, y_test, verbose=0)
score1b = model1b.evaluate(x_test, y_test, verbose=0)
score1c = model1c.evaluate(x_test, y_test, verbose=0)
score3 = model3.evaluate(x_test, y_test, verbose=0)

score2a0 = model2.evaluate(x_train, y_train, verbose=0)
score2b0 = model2b.evaluate(x_train, y_train, verbose=0)
score2c0 = model2c.evaluate(x_train, y_train, verbose=0)
score1a0 = model1.evaluate(x_train, y_train, verbose=0)
score1b0 = model1b.evaluate(x_train, y_train, verbose=0)
score1c0 = model1c.evaluate(x_train, y_train, verbose=0)
score30 = model3.evaluate(x_train, y_train, verbose=0)

print('Train loss 2a:', score2a0[0])
print('Train loss 2b:', score2b0[0])
print('Train loss 2c:', score2c0[0])
print('Train loss 1a:', score1a0[0])
print('Train loss 1b:', score1b0[0])
print('Train loss 1c:', score1c0[0])
print('Train loss 3:', score30[0])

print('Train accuracy 2a:', score2a0[1])
print('Train accuracy 2b:', score2b0[1])
print('Train accuracy 2c:', score2c0[1])
print('Train accuracy 1a:', score1a0[1])
print('Train accuracy 1b:', score1b0[1])
print('Train accuracy 1c:', score1c0[1])
print('Train accuracy 3:', score30[1])

print('Test loss 2a:', score2a[0])
print('Test loss 2b:', score2b[0])
print('Test loss 2c:', score2c[0])
print('Test loss 1a:', score1a[0])
print('Test loss 1b:', score1b[0])
print('Test loss 1c:', score1c[0])
print('Test loss 3:', score3[0])

print('Test accuracy 2a:', score2a[1])
print('Test accuracy 2b:', score2b[1])
print('Test accuracy 2c:', score2c[1])
print('Test accuracy 1a:', score1a[1])
print('Test accuracy 1b:', score1b[1])
print('Test accuracy 1c:', score1c[1])
print('Test accuracy 3:', score3[1])

print('Time 2a: ', stop2-start2)
print('Time 2b: ', stop2b-start2b)
print('Time 2c: ', stop2c-start2c)
print('Time 1a: ', stop1-start1)
print('Time 1b: ', stop1b-start1b)
print('Time 1c: ', stop1c-start1c)
print('Time 3: ', stop3-start3)
