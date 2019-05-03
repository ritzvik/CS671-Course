import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow import keras

import argparse
import numpy as np
import os
import glob
import cv2
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



def readData(path):
    imgs = []
    labels = []
    max1, max2 = 0,0
    Files = []

    
    files = [f for f in glob.glob(path + "Data/*.jpeg", recursive=True)]
    Files = [f for f in glob.glob(path + "Data/*.jpeg", recursive=True)]
    
    for f in files:
        im = cv2.imread(f,0)
        im = np.asarray(im)
        max1, max2 = max(max1, im.shape[0]), max(max2, im.shape[1])
        
        imgs.append(im)

        f = f.replace("Data", "Ground_truth" )
        f = f.replace(".jpeg", "_gt.txt")
    
        fi = open(f, "r")
        st = fi.read().split(" ")
        labels.append(np.asarray(st)[:2])
        

    print("Maximum Image size:", max1, max2)
    np.save("dims.npy", [max1, max2])

    img = []
    for i in imgs:
        i = np.asarray(i)
        i = np.pad(i,((0,max1-i.shape[0]),(0,max2-i.shape[1])), 'constant', constant_values=(0, 0))
        img.append(i)


    imgs = np.asarray(img, dtype=np.uint8)
    labels = np.asarray(labels, dtype=int)

    np.save("./imgs.npy", imgs)
    np.save("./labels.npy", labels)

    return imgs, labels, max1, max2, Files
    


def model(max1, max2):
    lines_input = Input(shape=(max1, max2,1), name='line')
    x = layers.Conv2D(32, (3, 3), activation='relu')(lines_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
    x = layers.Flatten()(x)

    reg = layers.Dense(100,activation='relu')(x)
    reg = layers.Dense(2, activation='linear', name='box')(reg)

    model = Model(lines_input,reg)

    model.compile(loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
    model.summary()

    return model

def train(X, Y, epoch, model, max1, max2):
    X = X.reshape(-1,max1,max2,1)
    
    history = model.fit(X,Y, 
                        epochs=epoch, 
                        batch_size=4,
                        verbose=1)

    model.save('q3.h5')

def test(X, Y, model, max1, max2, fs):

    model.load_weights("q3.h5")
    X = X.reshape(-1,max1,max2,1)
    print(X.shape)
    # score, acc = model.evaluate(X, Y, batch_size=16)

    # print("Accuracy:", acc )
    # print("Loss:", score)

    M = model.predict(X)
    try:
        os.mkdir("./results")
    except:
        pass
    for i in range(len(fs)):
        t = fs[i].split('/')
        try:
            os.mkdir("./results/"+t[len(t)-1].split('.')[0])
        except:
            pass
        f = open("./results/"+t[len(t)-1].split('.')[0]+"/pred.txt", "w")
        # f.write(np.array2string(M[i]))
        f.write(str(int(M[i][0]))+","+str(int(M[i][1])))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, help='Phase to train or test model on data, default is none.')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training, default is 1')
    args = parser.parse_args()

    if args.phase == "train":
        if args.epochs is None:
             epoch = 1
        else:
            epoch = args.epochs

        path = input("Enter the training folder:")
        X, Y, m1, m2, fs = readData(path)
        train(X, Y, epoch, model(m1, m2), m1, m2)
    elif args.phase == "test":
        path = input("Enter the training folder:")
        X, Y, m1, m2, fs = readData(path)
        test(X, Y, model(m1, m2), m1, m2, fs)
    else:
        print("Please mention the phase")
        

