import tensorflow as tf

def predict(path):
    model=load_model('model.h5')

    model.predict(path)
    