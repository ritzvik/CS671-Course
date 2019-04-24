from keras.models import *
from keras.layers import *
import keras
import cv2
import sys
import glob
import os

model = load_model('./my_model.h5')
dirname = sys.argv[1]

def readimg(filelist):
    data = list()
    for d in filelist:
        data.append(cv2.imread(d))
    return np.array(data, dtype=float)/255.0

datafiles = glob.glob(dirname+'/*')
D = readimg(datafiles)

names = [os.path.basename(x) for x in glob.glob(dirname+'/*')]
M = model.predict(D)

if not os.path.exists('./predicted_mask/'):
    os.makedirs('./predicted_mask/')

for m,name in zip(M[:,:,:,1],names):
    cv2.imwrite('./predicted_mask/'+name, m)
