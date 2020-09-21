import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from keras.models import load_model
import pyautogui

# Loading the model
model=load_model("model_drive.h5")
cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZERO',  1: 'FIVE'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    frame= cv2.resize(frame,(800,700))

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])-100
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction5
    roi = cv2.resize(roi, (400, 400))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    from keras.preprocessing import image
    x = image.img_to_array(test_image)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)

    # Batch of 1



    # print(x.shape)
    result = model.predict(x)



    if result[0][0]:
        cv2.putText(frame, "Center", (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 3)
        pyautogui.press('w')
        # cv2.imshow("Frame", frame)
    if result[0][1]:
        cv2.putText(frame, "Left", (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 3)
        pyautogui.press('a')
        # cv2.imshow("Frame", frame)
    if result[0][2]:
        cv2.putText(frame, "Right", (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 3)
        pyautogui.press('d')
    cv2.imshow("Frame", frame)
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc keyaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        break

cap.release()
cv2.destroyAllWindows()
