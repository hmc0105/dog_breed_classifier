import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from image_aug import *
import os
import pandas as pd
import shutil
import numpy as np

def train():
    current_path=os.getcwd()
    data_path=current_path+"/dataset"

    label_text=pd.read_csv((data_path+'/labels.csv'))

    train_sub_path=current_path+'/train_sub/'
    if not os.path.isdir(train_sub_path):
        os.mkdir(train_sub_path)

    for i in range(len(label_text)):
        if os.path.exists((train_sub_path+label_text.loc[i]['breed']))==False:
            os.mkdir((train_sub_path+label_text.loc[i]['breed']))
        shutil.copy((data_path+'/train/train/'+label_text.loc[i]['id']+'.jpg'),(train_sub_path+label_text.loc[i]['breed']))

    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


 #image generating
    image_size=299
    batch_size=32

    train_datagen=ImageDataGenerator(rescale=1./255.,horizontal_flip=True,shear_range=0.2,zoom_range=0.2,
    width_shift_range=0.2,height_shift_range=0.2,validation_split=0.25)
    valid_datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
 
    train_generator=train_datagen.flow_from_directory(directory=train_sub_path,subset="training",batch_size=batch_size,
    seed=42,shuffle=True,class_mode='categorical',target_size=(image_size,image_size))

    valid_generator=valid_datagen.flow_from_directory(directory=train_sub_path,subset='validation',batch_size=1,seed=42,
    shuffle=True,class_mode='categorical',target_size=(image_size,image_size))

    
    inceptionv3=InceptionV3()

    print(1)
    for layer in inceptionv3.layers[:-5]:
        layer.trainable=False
    for layer in inceptionv3.layers[-5:]:
        layer.trainable=True
    print(2)
    x=inceptionv3.layers[-2].output
    predictions=tf.keras.layers.Dense(120,activation='softmax')(x)
    model=tf.keras.Model(inputs=inceptionv3.input,outputs=predictions)

    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    print(5)
    history=model.fit_generator(generator=train_generator,steps_per_epoch=10,epochs=50,validation_data=valid_generator,validation_steps=3)
    print(3)

    model.save('model.h5')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],'b-',label='loss')
    plt.plot(history.history['val_loss'],'r--',label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],'g-',label='accuracy')
    plt.plot(history.history['val_accuracy'],'k--',label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0,0.1)

    plt.show()

