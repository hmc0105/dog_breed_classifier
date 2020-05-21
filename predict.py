import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

def predict(path):
    image_array=[]
    img=cv2.imread(path,1)
    org=img
    img=cv2.resize(img, dsize=(299,299))
    img=img/255.0
    image_array.append(img)
    image_array=np.array(image_array)
    model=tf.keras.models.load_model('model.h5')
    result=model.predict(image_array)[0].argsort()[::-1][:5]
    label_text=pd.read_csv('dataset/labels.csv')
    unique_Y=label_text['breed'].unique().tolist()
    unique_sorted_Y=sorted(unique_Y)
    top1='top1 : '+unique_sorted_Y[result[0]]
    top2='top2 : '+unique_sorted_Y[result[1]]
    top3='top3 : '+unique_sorted_Y[result[2]]
    top4='top4 : '+unique_sorted_Y[result[3]]
    top5='top5 : '+unique_sorted_Y[result[4]]
    cv2.putText(org, top1,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(org, top2,(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(org, top3,(0,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(org, top4,(0,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(org, top5,(0,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.imwrite("output.jpg",org)