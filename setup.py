import tensorflow as tf
import zipfile
import os

current_path=os.getcwd()
data_path=current_path+"/dataset"
if not os.path.isdir(data_path):
    os.mkdir(data_path)

tf.keras.utils.get_file((data_path+'/labels.csv'), 'http://bit.ly/2GDxsYS')
tf.keras.utils.get_file(data_path+'/sample_submission.csv', 'http://bit.ly/2GGnMNd')
tf.keras.utils.get_file(data_path+'/train.zip', 'http://bit.ly/31nIyel')
tf.keras.utils.get_file(data_path+'/test.zip', 'http://bit.ly/2GHEsnO')

train_zip=data_path+'/train.zip'
train_path=data_path+'/train'
print(train_zip)

#For some case, train_zip is not identified as Zipfile. So Pull out files by yourself
zip=zipfile.ZipFile(train_zip)
zip.extractall(train_path)
zip.close()

test_zip=data_path+'/test.zip'
test_path=data_path+'/test'
zip=zipfile.ZipFile(test_zip)
zip.extractall(test_path)
zip.close()
