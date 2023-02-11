import cv2
from matplotlib import image
import numpy as np

def nothing(x):
    pass

def rgb_to_hsv(r, g, b):
 
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
    # compute v
    v = cmax * 100
    return h, s, v

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

def mouseHSV(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        Hue = img_hsv[y,x,0]
        Stand = img_hsv[y,x,1]
        Value = img_hsv[y,x,2]
        colors = img_hsv[y,x]
        print("Hue: ",Hue)
        print("Stand: ",Stand)
        print("Value: ",Value)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        # h,s,v = rgb_to_hsv(colorsR,colorsG,colorsB)
        # print("Hue: ", h)
        # print("Sat: ", s)
        # print("Val: ", v)
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
    k = cv2.waitKey(1) & 0xFF  # checks the input - esc key
    if k == 27:
        break
