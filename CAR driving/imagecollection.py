import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("Datasets1"):
    os.makedirs("Datasets1")
    os.makedirs("Datasets1/Train")
    os.makedirs("Datasets1/Test")
    os.makedirs("Datasets1/Train/LT")
    os.makedirs("Datasets1/Test/L")
    os.makedirs("Datasets1/Train/RT")
    os.makedirs("Datasets1/Test/R")
    os.makedirs("Datasets1/Train/CT")
    os.makedirs("Datasets1/Test/C")



# Train or test
mode_train = 'Train'
directory_train = 'Datasets1/'+mode_train+'/'

mode_test = 'Test'
directory_test = 'Datasets1/'+mode_test+'/'

cap=cv2.VideoCapture(0)

count_L = 0
count_R = 0
count_C=0
# Collect 100 samples of your face from webcam input

while True:

    _, frame = cap.read()
    frame=cv2.flip(frame,1)
    frame =cv2.resize(frame,(800,700))

    count__train = {'left_train': len(os.listdir(directory_train+"/LT")),
                    'right_train': len(os.listdir(directory_train+"/RT")),
                    'center_train': len(os.listdir(directory_train+"/CT"))}

    count__test = {'left_test': len(os.listdir(directory_test+"/L")),
                   'right_test': len(os.listdir(directory_test+"/R")),
                   'center_test': len(os.listdir(directory_test+"/C"))}

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
    roi = cv2.resize(roi, (400, 400))

    cv2.imshow("Frame", frame)
#    120,255
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    interrupt=cv2.waitKey(10)


    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('1'):
        if count_L<=500:
            cv2.imwrite(directory_train+'LT/'+str(count__train['left_train'])+'.jpg', roi)

        elif count_L>500 and count__test['left_test']<=100:
            cv2.imwrite(directory_test+'L/'+str(count__test['left_test'])+'.jpg', roi)

        if count_L > 600:
            print("Completed Completed: ")

        count_L = count_L + 1


    if interrupt & 0xFF == ord('2'):
        if count_R<=500:
            cv2.imwrite(directory_train+'RT/'+str(count__train['right_train'])+'.jpg', roi)

        elif count_R>500 and count__test['right_test']<=100:
            cv2.imwrite(directory_test+'R/'+str(count__test['right_test'])+'.jpg', roi)

        if count_R > 600:
            print("Completed Completed: ")

        count_R = count_R + 1


    if interrupt & 0xFF == ord('3'):

        if count_C<=500:
            cv2.imwrite(directory_train+'CT/'+str(count__train['center_train'])+'.jpg', roi)

        elif count_C>500 and count__test['center_test']<=100:
            cv2.imwrite(directory_test+'C/'+str(count__test['center_test'])+'.jpg', roi)

        if count_C > 600:
            print("Completed Completed: ")

        count_C = count_C + 1


    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break





cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
