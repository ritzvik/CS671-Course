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


total_images = []
total_label = []
box_label=[]
f= open("./test/groundtruth.txt","r")
f1 = f.readlines()
for x in f1:
    columns = x.split(',')

    if(os.path.exists("./test/%s"%(columns[0]))):
        # print(columns[0])
        image = cv2.imread ("./test/%s"%(columns[0]),0)
        image = np.pad(image,((0,480-image.shape[0]),(0,640-image.shape[1])), 'constant', constant_values=(0, 0))
       	# print(image.shape)
        # resized_image = cv2.resize(image,(300,300))
        total_images.append (image.reshape(480,640,1))
        total_label.append((1,0,0))
        a=float(columns[1])
        b=float(columns[2])
        c=float(columns[3])
        d=float(columns[4])
        box_label.append((a,b,c,d))
# f= open("./test/groundtruth.txt","r")
# f1 = f.readlines()
# for x in f1:
#     columns = x.split(',')
#     if(os.path.exists("./test/%s"%(columns[0]))):
#         image = cv2.imread ("./test/%s"%(columns[0]),0)
#         # resized_image = cv2.resize(image,(300,300))
#         image = np.pad(image,((0,480-image.shape[0]),(0,640-image.shape[1])), 'constant', constant_values=(0, 0))
#         total_images.append (image.reshape(480, 640, 1))
#         total_label.append((0,1,0))
#         a=float(columns[1])
#         b=float(columns[2])
#         c=float(columns[3])
#         d=float(columns[4])
#         box_label.append((a,b,c,d))
# f= open("./test/groundtruth.txt","r")
# f1 = f.readlines()
# for x in f1:
#     columns = x.split(',')
#     if(os.path.exists("./test/%s"%(columns[0]))):
#         image = cv2.imread ("./test/%s"%(columns[0]),0)
#         image = np.pad(image,((0,480-image.shape[0]),(0,640-image.shape[1])), 'constant', constant_values=(0, 0))
#         # resized_image = cv2.resize(image,(300,300)) 
#         total_images.append (image.reshape(480, 640, 1))
#         total_label.append((0,0,1))
#         a=float(columns[1])
#         b=float(columns[2])
#         c=float(columns[3])
#         d=float(columns[4])
#         box_label.append((a,b,c,d))

total_images = np.array(total_images,dtype=np.uint8)
total_label = np.array(total_label)
box_label = np.array(box_label)

model = load_model("q1.h5")
pred = model.predict(total_images)
f= open("q1_pred.txt","w+")

def bb_intersection_over_union(boxA, boxB):
	
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0],
	 boxB[0])
	yA = max(boxA[1],
	 boxB[1])
	xB = min(boxA[2],
	 boxB[2])
	yB = min(boxA[3],
	 boxB[3])
 
	# compute the area of intersection rectangle
	if((xB - xA)==0 or (yB - yA)==0):
		iou = 0.0
		return iou
	interArea =  max(0, xB - xA + 1) * max(0, yB - yA + 1)                                                                                                                                                                                                                                                             
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

# print(box_label)
for i in range(total_images.shape[0]):
	pred_box=[]
	for x in range(4):
		if(x==0):
			pred_class = pred[0][i]
		pred_box.append(pred[1][i][x])
	pred_box=np.array(pred_box, dtype=float)
	f.write("%f %f %f %f %f %f %f %f %f %f %f %f \n" %(pred_class[0],
		pred_class[1],
		pred_class[2],
		pred_box[0],
		pred_box[1],
		pred_box[2],
		pred_box[3],
		box_label[i][0],
		box_label[i][1],
		box_label[i][2],
		box_label[i][3],
		bb_intersection_over_union(pred_box,box_label[i])
	))