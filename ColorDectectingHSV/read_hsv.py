import cv2
from matplotlib import image
import numpy as np
import easygui
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",600,250)
cv2.createTrackbar("Huemin", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Standmin", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Valuesmin", "Trackbars", 109, 255, nothing)
cv2.createTrackbar("Huemax", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("Standmax", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Valuesmax", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("BlobSize", "Trackbars", 0, 0, nothing)
cv2.createTrackbar("BlobSizes", "Trackbars", 0, 100, nothing)
import time

last_time = None
def mouseHSV(event,x,y,flags,param):
    global last_time
    if event == cv2.EVENT_LBUTTONDOWN: 
        if last_time is not None and time.time() - last_time < 1:
            Hue = img_hsv[y,x,0]
            Stand = img_hsv[y,x,1]
            Value = img_hsv[y,x,2]
            # colors = img_hsv[y,x]
            mess = 'HSV: {},{},{}'.format(Hue,Stand,Value)
            easygui.msgbox(mess)
            last_time = None
        else:  
            last_time = time.time()
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',mouseHSV)
while True:
    image = cv2.imread('test/smoke_(8950).jpg',)
    image = cv2.resize(image,(720,480))
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("Huemin", "Trackbars")
    l_s = cv2.getTrackbarPos("Standmin", "Trackbars")
    l_v = cv2.getTrackbarPos("Valuesmin", "Trackbars")
    u_h = cv2.getTrackbarPos("Huemax", "Trackbars")
    u_s = cv2.getTrackbarPos("Standmax", "Trackbars")
    u_v = cv2.getTrackbarPos("Valuesmax", "Trackbars")
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    height, width = mask.shape
    mask_find = np.zeros((height,width))
    # print(l_h,l_s,l_v,u_h,u_s,u_v)
    
    cv2.imshow("mask",mask)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  
    # for i in range(len(contours)):       
    #     area = cv2.contourArea(contours[i])    
    #     minContourSize = cv2.getTrackbarPos("BlobSize", "Trackbars")     
    #     if area > minContourSize :
    #         cv2.drawContours(image, contours, i, (0,0,0), 1, cv2.LINE_8, hierarchy, 0)
   
    color = (255,255,255)
    if contours:
        c = max(contours,key=cv2.contourArea)
        cv2.fillPoly(mask_find, pts=[c], color=color)
        cv2.drawContours(img_hsv,[c],0,1)
        cv2.imshow("frame", img_hsv)
        cv2.fillPoly(mask_find, pts=[c], color=color)
        cv2.imshow("mask",mask_find)
        cv2.drawContours(image, [c],0, 1)
    cv2.imshow("frame", image)
    cv2.imshow("frame", image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
