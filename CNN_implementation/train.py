# This is a demo program for construction of the CNN model in the paper:
# Qingwei Fang, Tiange Li, Yang Yang.2018.Predicting gene regulatory interactions of Drosophila eye development based on spatial gene expression data using deep learning. ISMB


from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.optimizers import adam, Adadelta,adagrad,SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import load_data
import csv

data_train, label_train, data_valid, label_valid ,data_test,label_test = load_data()
print(data_train.shape[0],'train samples')
print(data_valid.shape[0],'valid samples')
print(data_test.shape[0],'test samples')

#******************************************************************
model = Sequential()

model.add(Conv2D(filters = 4,kernel_size = (9,9), strides = 1, padding = "valid",input_shape = data_test.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 8,kernel_size=(7,7),padding = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 8,kernel_size=(5,5),padding = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 16,kernel_size=(3,3), padding = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 16,kernel_size=(3,3), padding = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 32,kernel_size = (3,3), padding = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#--------------------------------------------------------
model.add(Flatten())
model.add(Dense(128,kernel_initializer='normal'))
model.add(Activation('relu'))

model.add(Dense(1,kernel_initializer = 'normal'))
model.add(Activation('sigmoid'))

#*********************************************************
model.compile(loss = 'binary_crossentropy', optimizer='adagrad' ,metrics = ['accuracy'])
model.summary()
nb_epochs = 60
hist = model.fit(data_train,label_train,batch_size=16,epochs = nb_epochs,shuffle=True,verbose=1,validation_data=(data_valid, label_valid))

print ('test set outcome: ',model.evaluate(data_test,label_test,batch_size = 16))

# to save the training history
with open('training_proc_my_cnn.txt','w') as f:
    f.write(str(hist.history))
f.close()

# to save weights
model.save_weights('my_cnn-{}epochs_weights.h5'.format(nb_epochs))

# to make accuracy and F1 statistics based on edges rather than samples
predictions = model.predict(data_test)
test_csv = 'samples_test_mixed_set_lateral.csv'
old_file = []
with open(test_csv, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        new_row = [row[0], row[1], row[2]]
        old_file.append(new_row)
f.close()

# to store predictions into a csv file
with open('predictions_my_cnn.csv','w') as f:
    i = 0
    for ele in old_file:
        f.write(str(ele[0])+','+str(ele[1])+','+str(ele[2])+','+str(predictions[i])+'\n')
        i += 1
f.close()

# to calculate the accuracy and f1 score of predictions
from merge import evaluate_predictions
evaluate_predictions('predictions_my_cnn.csv')



