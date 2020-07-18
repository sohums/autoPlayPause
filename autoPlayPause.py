import cv2
import sys
import logging as log
from collections import deque
from pynput.mouse import Button, Controller
from time import sleep

cascPath = 'haarcascade_eye.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

vid_cap = cv2.VideoCapture(0)

buff_len = 30
buffer = deque(maxlen=buff_len)
for i in range(buff_len):
    buffer.append(1)

face_in_front = True
mouse = Controller()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    if not vid_cap.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = vid_cap.read()
    rescaled_frame = rescale_frame(frame, percent=50)

    gray = cv2.cvtColor(rescaled_frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(rescaled_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', rescaled_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Roughly 360 pings/minute (6 pings/second)
    buffer.append(len(faces))
    prev_fif = face_in_front
    face_in_front = sum([num > 0 for num in buffer]) > buff_len // 2

    if prev_fif != face_in_front:
        log.info('State change detected')
        mouse.position = (725, 375)

        mouse.press(Button.left)
        mouse.release(Button.left)

    log.info('{} : {}'.format(str(buffer), str(face_in_front)))

# When everything is done, release the capture
vid_cap.release()
cv2.destroyAllWindows()
