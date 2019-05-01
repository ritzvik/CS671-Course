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

image_path_1 = get_all_file_paths('./Four_Slap_Fingerprint/Image')
file_path_1  = get_all_file_paths('./Four_Slap_Fingerprint/Ground_truth')
image_path_1 = np.array(image_path_1)
def generate_data(image_path_1,batch_size):
  i = 0
  while True:
    total_images = []
    total_label=[]
    while (len(total_images)<batch_size):
      if (i == len(image_path_1)):
        i = 0
      np.random.shuffle(image_path_1)
      sample = image_path_1[i]
      i += 1
      c = sample.split('/')
      c = c[3].split('.')
      c=c[0]
      partial_labels=[] 
#       print(sample)
      if(os.path.exists("./Four_Slap_Fingerprint/Ground_truth/%s.txt"%(c))):
#         print("he")
        image = cv2.imread ("%s"%(sample),0)
        resized_image = cv2.resize(image,(500,500)) 
        #       resized_image = resized_image/255;
        total_images.append(resized_image.reshape(500, 500, 1))
#         print(len(total_images))

        f= open("./Four_Slap_Fingerprint/Ground_truth/%s.txt"%(c),"r")
        f1 = f.readlines()
        for x in f1:

          columns = x.split(',')
          col = columns[3].split('\n')
          a=int(columns[0])*(500/1572)
          b=int(columns[1])*(500/1672)
          c=int(columns[2])*(500/1572)
          d=int(col[0])*(500/1672)
          # col = columns[3].split('\n')
          # a=float(columns[0])
          # b=float(columns[1])
          # c=float(columns[2])
          # d=float(col[0])
          partial_labels.append((b,a,d,c))

        total_label.append((partial_labels[0],partial_labels[1],partial_labels[2],partial_labels[3]))

    total_images = np.array(total_images)
    total_label = np.array(total_label)
    total_label = np.reshape(total_label,(total_label.shape[0],(total_label.shape[1]*total_label.shape[2])))
    yield (total_images, {'box':total_label})
'''

total_images = []
total_label = []

for file_name in image_path_1:
    c = file_name.split('/')
    c = c[3].split('.')
    c=c[0]
    partial_labels=[]
    if(os.path.exists("./q2/Four_Slap_Fingerprint/Ground_truth/%s.txt"%(c))):
      
      image = cv2.imread ("%s"%(file_name))
      resized_image = cv2.resize(image,(300,300)) 
#       resized_image = resized_image/255;
      total_images.append(image)
      
      f= open("./q2/Four_Slap_Fingerprint/Ground_truth/%s.txt"%(c),"r")
      f1 = f.readlines()
      o=0
      for x in f1:
            
        columns = x.split(',')
        col = columns[3].split('\n')
        a=int(columns[0])*(300/1572)
        b=int(columns[1])*(300/1672)
        c=int(columns[2])*(300/1572)
        d=int(col[0])*(300/1672)
        # col = columns[3].split('\n')
        # a=float(columns[0])
        # b=float(columns[1])
        # c=float(columns[2])
        # d=float(col[0])
        partial_labels.append((a,b,c,d))
      
      total_label.append((partial_labels[0],partial_labels[1],partial_labels[2],partial_labels[3]))
'''

lines_input = Input(shape=(500, 500, 1), name='line')
x = layers.Conv2D(32, (3, 3), activation='relu')(lines_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
# x = layers.Conv2D(32, (3, 3), activation='relu')(x)

x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(128, (3, 3), activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(16, (3, 3), activation='relu', strides=2)(x)
# x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(4, 4),strides=2)(x)
x = layers.Flatten()(x)

reg = layers.Dense(512,activation='relu')(x)
reg = layers.BatchNormalization()(reg)
reg = layers.Dense(256,activation='relu')(reg)
reg = layers.BatchNormalization()(reg)
reg = layers.Dense(16, name='box', activation='linear')(x)


model = Model(lines_input,[reg])

model.compile(loss={'box':'mean_squared_error'},
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
# print("fweffwefgrgteg")
bs=16
history = model.fit_generator(generate_data(image_path_1,bs),
					steps_per_epoch=len(image_path_1)//bs,
                    epochs=10,
                    validation_data=generate_data(image_path_1, bs),
                    validation_steps=1,
                    verbose=1)
# history = model.fit(total_images,total_label, 
#                     epochs=3	, 
#                     validation_split=0.1,
#                     batch_size=32,
#                     verbose=1)
# y_pred = model.predict(test_images)
model.save("q2.h5")