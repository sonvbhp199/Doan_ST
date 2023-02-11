import os
import cv2
import shutil
path = "lamlaimask_512"
test_y = os.listdir(path)
from imutils.video import VideoStream
# video = cv2.VideoCapture("test.mp4")
# print(video.get(cv2.CAP_PROP_FPS))
# ret, frame = video.read()
# while ret==True:
#     ret, frame = video.read()
#     cv2.imshow("",frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
