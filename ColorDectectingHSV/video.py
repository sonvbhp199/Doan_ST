# tach frame
import cv2
import numpy as np
cap = cv2.VideoCapture('smoke_32.mp4')
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps

second = 0
cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
success, image = cap.read()
while success and second <= duration:
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, image = cap.read()
    cv2.imwrite("test/smoke_({}).jpg".format(str(second+8961)),image)
    second+=1