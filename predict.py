import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

def predict(path):
    image_array=[]
    img=cv2.imread(path)
    img=cv2.resize(img, dsize=(299,299))
    img=img/255.0
    image_array.append(img)
    image_array=np.array(image_array)
    model=tf.keras.models.load_model('model.h5')
    result=model.predict(image_array)[0].argsort()[::-1][:5]
    label_text=pd.read_csv('dataset/labels.csv')
    unique_Y=label_text['breed'].unique().tolist()
    unique_sorted_Y=sorted(unique_Y)
    print('top1 : ',unique_sorted_Y[result[0]])
    print('top2 : ',unique_sorted_Y[result[1]])
    print('top3 : ',unique_sorted_Y[result[2]])
    print('top4 : ',unique_sorted_Y[result[3]])
    print('top5 : ',unique_sorted_Y[result[4]])