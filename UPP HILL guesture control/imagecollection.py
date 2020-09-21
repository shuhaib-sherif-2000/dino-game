import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("Datasets1"):
    os.makedirs("Datasets1")
    os.makedirs("Datasets1/Train")
    os.makedirs("Datasets1/Test")
    os.makedirs("Datasets1/Train/0L")
    os.makedirs("Datasets1/Test/0L")
    os.makedirs("Datasets1/Train/0R")
    os.makedirs("Datasets1/Test/0R")
    os.makedirs("Datasets1/Train/5L")
    os.makedirs("Datasets1/Test/5L")
    os.makedirs("Datasets1/Train/5R")
    os.makedirs("Datasets1/Test/5R")


# Train or test
mode_train = 'Train'
directory_train = 'Datasets1/'+mode_train+'/'

mode_test = 'Test'
directory_test = 'Datasets1/'+mode_test+'/'


cap = cv2.VideoCapture(0)

count_0 = 0
count0=0
count_5 = 0
count5=0


while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    count__train = {'zero_trainL': len(os.listdir(directory_train+"/0L")),
                    'five_trainR': len(os.listdir(directory_train+"/5R")),
                    'zero_trainR': len(os.listdir(directory_train+"/0R")),
                    'five_trainL': len(os.listdir(directory_train+"/5L"))}

    count__test = {'zero_testL': len(os.listdir(directory_test+"/0L")),
                   'five_testR': len(os.listdir(directory_test+"/5R")),
                   'zero_testR': len(os.listdir(directory_test+"/0R")),
                   'five_testL': len(os.listdir(directory_test+"/5L"))}

    # Getting count of existing images

    # Coordinates of the ROI

    x=int(0.5*frame.shape[1])+80
    y=70
    x_=frame.shape[1]-10
    y_=int(0.5*frame.shape[1])-30
    cv2.rectangle(frame, (x-1, y-1), (x_+1, y_+1), (255,0,0) ,1)

    roi = frame[y:y_, x:x_]
    roi = cv2.resize(roi, (224, 224))
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)


    x1 = int(0.5*frame.shape[1])-312
    y1 = 70
    x2 = frame.shape[1]-400
    y2 = int(0.5*frame.shape[1])-30
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi1 = frame[y1:y2, x1:x2]
    roi1 = cv2.resize(roi1, (224, 224))
    # do the processing after capturing the image!
    roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    _, roi1 = cv2.threshold(roi1, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI1", roi1)
    #frame=cv2.resize(frame,(1000,1000))
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break

    if interrupt & 0xFF == ord('0'):

        if count_0<=500:

            cv2.imwrite(directory_train+'0L/'+str(count__train['zero_trainL'])+'.jpg', roi1)




        elif count_0>500 and count__test['zero_testL']<=100:

            cv2.imwrite(directory_test+'0L/'+str(count__test['zero_testL'])+'.jpg', roi1)


        if count_0 > 600:
            print("Completed Completed: ")

        count_0 = count_0 + 1


    if interrupt & 0xFF == ord('1'):

        if count0<=500:
            cv2.imwrite(directory_train+'0R/'+str(count__train['zero_trainR'])+'.jpg', roi)

        elif count0>500 and count__test['zero_testR']<=100:
            cv2.imwrite(directory_test+'0R/'+str(count__test['zero_testR'])+'.jpg', roi)

        if count0 > 600:
            print("Completed Completed: ")

        count0 = count0 + 1

        # print("Count of 0 commpleted: " + str(count_0))


    # print("Count of 0 commpleted: ")



    if interrupt & 0xFF == ord('5'):


        if count_5<=500:

            cv2.imwrite(directory_train+'5L/'+str(count__train['five_trainL'])+'.jpg', roi1)



        elif count_5>500 and count__test['five_testL'] <=100:

            cv2.imwrite(directory_test+'5L/'+str(count__test['five_testL'])+'.jpg', roi1)


        if count_5 > 600:
            print("Completed completed: ")

        count_5 = count_5 + 1



    if interrupt & 0xFF == ord('6'):
        if count5<=500:
            cv2.imwrite(directory_train+'5R/'+str(count__train['five_trainR'])+'.jpg', roi)

        elif count5>500 and count__test['five_testR'] <=100:
            cv2.imwrite(directory_test+'5R/'+str(count__test['five_testR'])+'.jpg', roi)

        if count5 > 600:
            print("Completed completed: ")

        count5 = count5 + 1

    # print("Count of 5 completed: ")




cap.release()
cv2.destroyAllWindows()
