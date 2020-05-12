import os
import pandas as pd
import shutil
import numpy as np

def image_aug():
    # 1. Separate files to escape overloading
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


    #Formation of np.array

    train_X=[]
    train_Y=[]

    for idx in range(7718):
        if idx%10==0:
            print(idx)
        x,y=train_generator.next()
        train_X.extend(x)
        train_Y.extend(y)
    train_X=np.array(train_X)
    train_Y=np.array(train_Y)

    valid_X=[]
    valid_Y=[]
    for idx in range(2504):
        if idx%200==0:
            print(idx)
        x,y=valid_generator.next()
        valid_X.extend(x)
        valid_Y.extend(y)
    valid_X=np.array(valid_X)
    valid_Y=np.array(valid_Y)

    return train_X,train_Y,valid_X, valid_Y