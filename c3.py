import cv2 as cv
import subprocess
import sys

def face_detect(input_image_name, output_image_name):
    ''' This function takes an input photo name
    and an output photo name and distinguishes the face
    from the input photo and draws a square around the face
    with the input color and saves it with the output photo name. '''
    faceCascade = cv.CascadeClassifier("Face Detect\\haarcascades\\haarcascade_frontalface_default.xml")

    img_color = cv.imread(input_image_name)
    img_gray  = cv.cvtColor(img_color,cv.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(img_gray,scaleFactor=1.01,minNeighbors = 5)

    if len(face)>0: 
        (x,y,w,h) = face[0]
        our_image_rect = cv.rectangle(img_color,(x,y),(x+w,y+h),(0,0,255),2)
        cv.imwrite(output_image_name , our_image_rect)
        cv.imshow("face detect",our_image_rect)

face_detect("e4.jpg","e4-det.jpg")
