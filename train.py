import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from image_aug import *

def train():
    train_X,train_Y,valid_X,valid_Y=image_aug()
    print(train_X)
    inceptionv3=InceptionV3()

    print(1)
    for layer in inceptionv3.layers[:-10]:
        layer.trainable=False
    for layer in inceptionv3.layers[-10:]:
        layer.trainable=True
    print(2)
    x=inceptionv3.layers[-2].output
    predictions=tf.keras.layers.Dense(120,activation='softmax')(x)
    model=tf.keras.Model(inputs=inceptionv3.input,outputs=predictions)

    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print(5)
    hisotry=model.fit(train_X,train_Y,epochs=10,validation_split=0.25,batch_size=32)
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

