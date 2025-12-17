
import cv2
import numpy as np
img = cv2.imread("img.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_x = np.array([
[-1, 0, 1],
[-1, 0, 1],
[-1, 0, 1]
], dtype=np.float32)

kernel_y= np.array( [
[-1, -1, -1],
[ 0, 0, 0],
[ 1, 1, 1]
], dtype=np.float32)

gx = cv2.filter2D (gray, -1, kernel_x)
gy= cv2.filter2D (gray, -1, kernel_y)

prewitt = cv2.addWeighted(cv2.convertScaleAbs (gx),  0.5,
                          cv2.convertScaleAbs(gy), 0.5, 0)
result = 255 - prewitt

cv2.imshow( "Prewitt (oq-gora)", result)
cv2.waitKey(0)
