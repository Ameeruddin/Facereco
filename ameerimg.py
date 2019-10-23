# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:57:26 2019

@author: Mohammed
"""

import cv2
cap=cv2.VideoCapture(0)
image_counter=0
while (True):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    if not ret:
        break
    k=cv2.waitKey(1)
    if k%256==27:
        print("thanks")
        break
    elif k%256==32:
        img_name="image_{}.png".format(image_counter)
        cv2.imwrite(img_name,frame)
        print("{} written".format(image_counter))
        image_counter+=1
cap.release()
cv2.destroyAllWindows()
