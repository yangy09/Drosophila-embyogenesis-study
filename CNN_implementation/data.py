import os
import csv
import numpy as np
from PIL import Image
from keras.utils import np_utils, generic_utils


def read_image(img_url):

    img = Image.open(img_url).convert('L')
    arr1 = np.asarray(img,"float64")

    return arr1


def data_set_loading(samples_csv):

    # dir_path is where FlyExpress ISH image data is stored.
    dir_path = '/data/flyexpress/data/pic_data/'

    samples =[]

    # csv_path is where 'samples_train_mixed_set_lateral.csv', 'samples_valid_mixed_set_lateral.csv', and 'samples_test_mixed_set_lateral.csv' are stored.
    csv_path = '/data/code/' + samples_csv
    print(csv_path)
    with open(csv_path,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            new_row = [row[0],row[1],row[2]]
            samples.append(new_row)
    f.close()

    len_of_samples = len(samples)
    data = np.empty((len_of_samples,256,320,1),dtype = "float64")
    label = np.empty((len_of_samples,),dtype = "int")

    #-----------------------------------------------------------
    for i in range(len_of_samples):

        row = samples[i]
        img1 = read_image(dir_path + str(row[0]).split('&')[-1])
        img2 = read_image(dir_path + str(row[1]).split('&')[-1])
        data[i,0:128,:,0] = img1
        data[i,128:256,:,0] = img2

        label0 = int(row[2])
        label[i] = label0

        info = 'image ' + str(i) + ' has been read.   Progress: ' + str(((i + 1) * 100) / len_of_samples) + '%'
        print(info)


    return data,label


def load_data():

    train_csv = 'samples_train_mixed_set_lateral.csv'
    valid_csv = 'samples_valid_mixed_set_lateral.csv'
    test_csv = 'samples_test_mixed_set_lateral.csv'

    data_train, label_train = data_set_loading(train_csv)
    data_valid,label_valid = data_set_loading(valid_csv)
    data_test, label_test = data_set_loading(test_csv)

    data_train /= 255.0
    data_valid /= 255.0
    data_test /= 255.0

    return data_train,label_train,data_valid,label_valid,data_test,label_test

