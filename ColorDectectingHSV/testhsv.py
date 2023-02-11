import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from cv2 import cv2

blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
lo_square = np.full((10, 10, 3), hsv_blue, dtype=np.uint8) / [180., 255., 255.]
print(lo_square)
plt.subplot(1, 1, 1)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()