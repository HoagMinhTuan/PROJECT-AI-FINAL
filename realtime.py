from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys

model5=load_model("E:/AI cuoi ky/bao cao/Project AI Final.h5")

cap = cv2.VideoCapture("E:/AI cuoi ky/code/Camera/xoai.mp4")
class_name =['Xoai']
while(True):
    # Capture frame-by-frame
    #
    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None,fx=0.4,fy=0.4)
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(300,300))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)


    # pic = image.load_img('../BAOCAOCUOIKI/test/asagi2.jpg', target_size=(250,200))
    # plt.imshow(pic)
    # pic = img_to_array(pic)
    # pic = pic.reshape(1,250,200,3) 
    # pic = pic.astype('float32')
    # pic = pic/255
    # print(np.argmax(model.predict(pic), axis=1))
    # print(labels)

    # Predict
    #predict = model5.predict(image)
    # print("This picture is: ", class_name[np.argmax(predict[0])])
    #print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    #print(np.max(predict[0],axis=0))
    #if (np.max(predict)>=0.7) and (np.argmax(predict[0])!=0):


        # # Show image
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # org = (50, 50)
        # fontScale = 1
        # color = (0, 255, 0)
        # thickness = 2

        # cv2.putText(image_org, class_name[np.argmax(predict)],org, font,
        #              fontScale, color, thickness, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    text = 'Xoai'
    cv2.putText(image_org,text,org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()