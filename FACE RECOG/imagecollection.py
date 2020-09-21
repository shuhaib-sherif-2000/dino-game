import cv2
import numpy as np
import os


# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = input("Enter your name: ")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# Example
createFolder('./Datasets/Test/{}/'.format(name))
createFolder('./Datasets/Train/{}/'.format(name))

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)


    if faces is ():
        return None


    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]



    return cropped_face



# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0


# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()


    if face_extractor(frame) is not None:

        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        # face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


        # Save file in specified directory with unique name
        if count<=500:

            file_name_path = './Datasets/Train/{}/{}.jpg'.format(name, str(count))
            cv2.imwrite(file_name_path, face)
            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)

        elif count>500:
            file_name_path = './Datasets/Test/{}/{}.jpg'.format(name, str(count))
            cv2.imwrite(file_name_path, face)
            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)




    else:

        print("Face not found")
        pass


    if cv2.waitKey(1) == 13 or count == 600: #13 is the Enter Key
        break



cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
