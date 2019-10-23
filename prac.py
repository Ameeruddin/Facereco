import cv2
import numpy
import mysql.connector
import matplotlib 
import pickle
import time


def fun1(label1):
    if(dicta[label1]!=-1):
        mycursor=mydb.cursor()
    
        print(str(label1))
    
        print(dicta)
        dicta[label1]=dicta[label1]+1
        time.sleep(1)
        if dicta[label1]>5 and dicta[label1]!=-1:
            dicta[label1]=-1
            val=('P',label1)
            result=mycursor.execute('UPDATE miniproject3 set STATUS=%s where ID=%s',val)
            mydb.commit()
    else:
        return 0



dicta={'090':0,'098':0}
mydb=mysql.connector.connect(host="localhost",database='attd1',user='root',password='')
face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml ')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
cap=cv2.VideoCapture(0)
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
while(True):
    qret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_g=gray[x:x+w,y:y+h]
        roi_c=frame[x:x+w,y:y+h]
        id_,conf=recognizer.predict(roi_g)
        if conf>80:
            #print(id_)
            print(str(labels[id_]))
            
            fun1(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            color=(255,255,255)
            name=labels[id_]
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
            img='my_image.png'    
            cv2.imwrite(img,roi_c)
            color=(255,0,0)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


    