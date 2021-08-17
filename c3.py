import cv2 as cv
import subprocess
import sys


def face_detect(input_image_name, output_image_name, mode):
    ''' This function takes an input photo name
    and an output photo name and distinguishes the face
    from the input photo and draws a square around the face
    with the input color and saves it with the output photo name. '''
    faceCascade = cv.CascadeClassifier("Face Detect\\haarcascades\\haarcascade_frontalface_default.xml")

    img_color = cv.imread(input_image_name)
    img_color_copy = cv.imread(input_image_name)
    img_gray  = cv.cvtColor(img_color,cv.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(img_gray,scaleFactor=1.01,minNeighbors = 5)

    if len(face)>0: 
        (x,y,w,h) = face[0]
        if mode == "circle":
            center_coordinates = (int(x+w/2),int(y+h/2))
            radius = int(w/2+5)
            color = (0,0,255)
            thickness = 2
            our_image_circle = cv.circle(img_color, center_coordinates, radius, color, thickness)
            cv.imwrite('c-'+output_image_name , our_image_circle)
            cv.imshow("face detect circle",our_image_circle)
        elif mode == "rectangle":
            our_image_rect = cv.rectangle(img_color_copy,(x,y),(x+w,y+h),(0,0,255),2)
            cv.imwrite('r-'+output_image_name , our_image_rect)
            cv.imshow("face detect rectangle",our_image_rect)
        else:
            print('mode is invalid')

face_detect("e2.jpg","e2.jpg","circle")
