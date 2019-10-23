# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:31:12 2019

@author: Mohammed
"""

import os
import cv2
from PIL import Image
import numpy as np
import pickle
train=[]
ylabel=[]
label_ids={}
current_id=0

face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
dire=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(dire,"images")
for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path))
            print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            pil_image=Image.open(path).convert("L")
            size=(550,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            
            img_array=np.array(final_image,"uint8")
            #print(img_array)
            faces=face_cascade.detectMultiScale(img_array,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=img_array[y:y+h,x:x+w]
                train.append(roi)
                ylabel.append(id_)                
#print(train)
#print(label)
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)
recognizer.train(train,np.array(ylabel))
recognizer.save("trainner.yml")
 
 
