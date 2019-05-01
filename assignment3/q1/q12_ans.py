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

image_path_1 = get_all_file_paths('./test/Image')
file_path_1  = get_all_file_paths('./test/Ground_truth')

total_images = []
total_label = []

for file_name in image_path_1:
    c = file_name.split('/')
    c = c[3].split('.')
    c=c[0]
    partial_labels=[]
    if(os.path.exists("./test/Ground_truth/%s.txt"%(c))):
      
      image = cv2.imread ("%s"%(file_name),0)
      resized_image = cv2.resize(image,(500,500)) 
#       resized_image = resized_image/255;
      total_images.append(resized_image.reshape(500,500,1))
      
      f= open("./test/Ground_truth/%s.txt"%(c),"r")
      f1 = f.readlines()
      o=0
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
        partial_labels.append((a,b,c,d))
      
      total_label.append((partial_labels[0],partial_labels[1],partial_labels[2],partial_labels[3]))



total_images = np.array(total_images,dtype=np.uint8)
total_label = np.array(total_label)
total_label = np.reshape(total_label,(total_label.shape[0],(total_label.shape[1]*total_label.shape[2])))

print(total_images.shape)
model = load_model("q2.h5")
pred = model.predict(total_images)
f= open("q2_pred.txt","w+")

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

# print(total_label)
for i in range(total_images.shape[0]):
	pred_box=[]
	for x1 in range(16):
		pred_box.append(pred[i][x1])
	pred_box=np.array(pred_box, dtype=float)
	f.write(" %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n" %(
		pred_box[0],
		pred_box[1],
		pred_box[2],
		pred_box[3],
		pred_box[4],
		pred_box[5],
		pred_box[6],
		pred_box[7],
		pred_box[8],
		pred_box[9],
		pred_box[10],
		pred_box[11],
		pred_box[12],
		pred_box[13],
		pred_box[14],
		pred_box[15],
		total_label[i][0],
		total_label[i][1],
		total_label[i][2],
		total_label[i][3],
		total_label[i][4],
		total_label[i][5],
		total_label[i][6],
		total_label[i][7],
		total_label[i][8],
		total_label[i][9],
		total_label[i][10],
		total_label[i][11],
		total_label[i][12],
		total_label[i][13],
		total_label[i][14],
		total_label[i][15],
		bb_intersection_over_union(pred_box[0:4],total_label[i][0:4]),
		bb_intersection_over_union(pred_box[4:8],total_label[i][4:8]),
		bb_intersection_over_union(pred_box[8:12],total_label[i][8:12]),
		bb_intersection_over_union(pred_box[12:16],total_label[i][12:16])
	))