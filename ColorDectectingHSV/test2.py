import cv2
from matplotlib import image
import numpy as np

def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",600,250)
cv2.createTrackbar("Huemin", "Trackbars", 0, 360, nothing)
cv2.createTrackbar("Standmin", "Trackbars", 39, 255, nothing)
cv2.createTrackbar("Valuesmin", "Trackbars", 6, 255, nothing)
cv2.createTrackbar("Huemax", "Trackbars", 62, 360, nothing)
cv2.createTrackbar("Standmax", "Trackbars", 173, 255, nothing)
cv2.createTrackbar("Valuesmax", "Trackbars", 247, 255, nothing)
cv2.createTrackbar("BlobSize", "Trackbars", 1000, 1000, nothing)

while True:
    image = cv2.imread('test/smoke_(7001).jpg')
    image = cv2.resize(image,(720,480))
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.resize(image,(720,480))
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
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  
    color = (255,255,255)
   
    # for i in range(len(contours)):
    #     c= max(contours,key=cv2.contourArea)
        # area = cv2.contourArea(contours[i])    
        # minContourSize = cv2.getTrackbarPos("BlobSize", "Trackbars")     
        # if area > minContourSize :
        #     # cv2.drawContours(image, contours, i, (0,0,0), 2, cv2.LINE_8, hierarchy, 0)
        #     cv2.fillPoly(mask_find, pts=[contours[i]], color=color)
    c = max(contours,key=cv2.contourArea)
    cv2.fillPoly(mask_find, pts=[c], color=color)
    cv2.drawContours(img_hsv,[c],0,1)
    cv2.imshow("frame", img_hsv)
    cv2.fillPoly(mask_find, pts=[c], color=color)
    cv2.imshow("mask",mask_find)
    cv2.imshow("frame", image)
    k = cv2.waitKey(1) & 0xFF  # checks the input - esc key
    if k == 27:
        break
