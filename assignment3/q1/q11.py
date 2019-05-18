# -*- coding: utf-8 -*-
"""q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10BkFklJz61t0bPO69BieTb8AmiC8GUtX
"""

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import os 
import cv2
import glob
from tensorflow import keras



def get_all_file_paths(directory): 
  
    # initializing empty file paths list 
    file_paths = [] 
  
    # crawling through directory and subdirectories 
    for root, directories, files in os.walk(directory): 
        for filename in files: 
            # join the two strings in order to form the full filepath. 
            filepath = os.path.join(root, filename) 
            file_paths.append(filepath) 
  
    # returning all file paths 
    return file_paths

f1= open("./Knuckle/groundtruth.txt","r")
f2= open("./Palm/groundtruth.txt","r")
f3= open("./Vein/groundtruth.txt","r")
f = np.array(f1.readlines()+f2.readlines()+f3.readlines())
# np.random.shuffle(f)
def generate_data(f,batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    while True:
        total_images = []
        total_label = []
        box_label=[]
        while (len(total_images)<batch_size):
            if i == len(f):
                i = 0
            np.random.shuffle(f)
            sample = f[i]
            i += 1
            columns = sample.split(',')            
            if(columns[5]=="vein\n"):
                  if(os.path.exists("./Vein/%s"%(columns[0]))):
                    print("here")
                    image = cv2.imread ("./Vein/%s"%(columns[0]),0)
                    image = np.pad(image,((0,480-image.shape[0]),(0,640-image.shape[1])), 'constant', constant_values=(0, 0))
                    # resized_image = cv2.resize(image,(300,300)) 
                    total_images.append (image.reshape(480, 640, 1))
                    total_label.append((0,0,1))
                    a=float(columns[1])
                    b=float(columns[2])
                    c=float(columns[3])
                    d=float(columns[4])
                    box_label.append((a,b,c,d))
          
            if(columns[5]=="knuckle\n"):
              if(os.path.exists("./Knuckle/%s"%(columns[0]))):
                  image = cv2.imread ("./Knuckle/%s"%(columns[0]),0)
                  image = np.pad(image,((0,480-image.shape[0]),(0,640-image.shape[1])), 'constant', constant_values=(0, 0))
                  # resized_image = cv2.resize(image,(300,300))
                  total_images.append (image.reshape(480,640,1))
                  total_label.append((1,0,0))
                  a=float(columns[1])
                  b=float(columns[2])
                  c=float(columns[3])
                  d=float(columns[4])
                  box_label.append((a,b,c,d))
        
            if(columns[5]=="palm\n"):
              if(os.path.exists("./Palm/%s"%(columns[0]))):
                  image = cv2.imread ("./Palm/%s"%(columns[0]),0)
                  # resized_image = cv2.resize(image,(300,300))
                  # image = np.pad(image,((0,480-image.shape[0]),(0,640-image.shape[1])), 'constant', constant_values=(0, 0))
                  total_images.append (image.reshape(480, 640, 1))
                  total_label.append((0,1,0))
                  a=float(columns[1])
                  b=float(columns[2])
                  c=float(columns[3])
                  d=float(columns[4])
                  box_label.append((a,b,c,d))
        total_images = np.array(total_images)
        total_label = np.array(total_label)
        box_label = np.array(box_label)
        yield (total_images, {'class': total_label, 'box':box_label})
        

lines_input = Input(shape=(480, 640, 1), name='line')
x = layers.Conv2D(32, (3, 3), activation='relu')(lines_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)


x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
x = layers.Flatten()(x)


classification = layers.Dense(100,activation='relu')(x)
classification = layers.Dense(50,activation='relu')(classification)
classification = layers.Dense(3, activation='softmax', name='class')(classification)

reg = layers.Dense(512,activation='relu')(x)
reg = layers.BatchNormalization()(reg)
reg = layers.Dense(256,activation='relu')(reg)
reg = layers.BatchNormalization()(reg)
reg = layers.Dense(4, name='box', activation='linear')(x)

model = Model(lines_input,[classification,reg])

model.compile(loss={'class': 'categorical_crossentropy',
                    'box':'mean_squared_error'},
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
print("fweffwefgrgteg")
bs=32
history = model.fit_generator(generate_data(f,bs),
					steps_per_epoch=len(f)//bs,
                    epochs=2,
                    validation_data=generate_data(f, bs),
                              validation_steps=1,
                    verbose=1)
model.save("q1.h5")
