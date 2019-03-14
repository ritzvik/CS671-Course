#%%
import numpy as np
import cv2
import os
from PIL import Image

#%%

def create_img(originx, originy, angle, len, color, thick):
    nimg = np.zeros([28,28,3],dtype=np.uint8)
    if len == 0:
        len = 7
    else:
        len = 15
    for i in range(0, len+1):
        x =  (originx + int(round(i*np.cos(angle))))
        y =  (originy - int(round(i*np.sin(angle))))
        if x < 0 or x >27 or y<0 or y>27:
            return 0, nimg

        if thick ==0:
            nimg[y][x][color] = 255

        elif angle == np.pi/2 and x > 1 and x < 26 :
            nimg[y][(x-1)][color] = 255
            nimg[y][x][color] = 255
            nimg[y][(x+1)][color] = 255

        elif angle != np.pi/2 and y > 1 and y < 26 :
            nimg[(y-1)][x][color] = 255
            nimg[y][x][color] = 255
            nimg[(y+1)][x][color] = 255

        else:
            return 0, nimg


    return 1, nimg

#%%

#%%

imlst = []
imhlst = []
ct = 0

try:
    os.mkdir("./video")
    os.mkdir("./images")
except:
  print("Directory already exists!")
  
    
out = cv2.VideoWriter('./video/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, (28*3,28*3))
new_im = Image.new('RGB', (28*3, 28*3))

#%%
imgh = np.zeros([28,28*3,3])

def build_video(img):
    X = ct%9
    Y = ct%3
    global imlst
    global imgh
    global imhlst
    global out
    imlst.append(img)
    if(Y == 2):
        imgh = np.hstack(i for i in imlst )
        imhlst.append(np.zeros([28,28*3,3])+imgh)
        imlst  = []

        if X == 8:
            imgv = np.vstack( i for i in imhlst )
            imhlst = []
                
            cv2.imwrite('video/' + '1' + '.jpg', imgv)
            image = cv2.imread('video/' + '1' + '.jpg')
            out.write(image)
            
     


#%%



for l in range(0,2): #len
    for t in range(0,2): #thick
        for a in range(0,12): #angle
            for c in range(0,3,2): #color
                try:
                    os.mkdir('./images/'+str(l)+'_' + str(t) + '_' + str(a) + '_' + str(int(c/2)))
                except:
                    print('./images/'+str(l)+'_' + str(t) + '_' + str(a) + '_' + str(int(c/2))+" already exists!")
                count = 1000
                while count > 0:
                    for r in range(0,28):
                        if count <= 0:
                            break
                        for d in range(0,28):
                            flag, img = create_img(d,r,(15*a)*np.pi/180, l,c, t)
                            if flag == 0:
                                continue
                            if count <=0:
                                break
                            
                            strn = str(l)+'_' + str(t) + '_' + str(a) + '_' + str(int(c/2)) + '_' + str(1001-count)
                            count-=1
                            
                            im = Image.fromarray(img)
                            im.save('images/'+str(l)+'_' + str(t) + '_' + str(a) + '_' + str(int(c/2)) +'/' + strn + '.jpg', quality = 100000)
                            # im.show()
                            if count >= 910:
                                build_video(img)
                                ct+=1

#%%
os.remove("./video/1.jpg")
out.release()