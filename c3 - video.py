import cv2
import subprocess
import sys

def face_detect(input_video_name,mode):
    ''' This function takes an input photo name
    and an output photo name and distinguishes the face
    from the input photo and draws a square around the face
    with the input color and saves it with the output photo name. '''
    faceCascade = cv2.CascadeClassifier("Face Detect\\haarcascades\\haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(input_video_name)

    video_width = int(cap.get(3))

    video_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    output_src = cv2.VideoWriter('detected-video.avi', fourcc, 30,(video_width,video_height))
    
    while True:
        ret, img = cap.read()
        if (type(img) == type(None)):
            break
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 4)

        for (x,y,w,h) in faces:
            if mode == "circle":
                center_coordinates = (int(x+w/2),int(y+h/2))
                radius = int(w/2+5)
                color = (0,0,255)
                thickness = 2
                our_image_circle = cv.circle(img, center_coordinates, radius, color, thickness)
            elif mode == "rectangle":
                our_image_rect = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            else:
                print('mode is invalid')

    
        cv2.imshow('video', img)
        if ret == True:
            output_src.write(img)
    
        if cv2.waitKey(1) == ord('q'):
            break
    output_src.release()
    cap.release()
    cv2.destroyAllWindows()
face_detect("a.mp4","circle")
