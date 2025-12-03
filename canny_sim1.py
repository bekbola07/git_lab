import cv2
img = cv2.imread("img_3.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur (gray,  (5,5),  1.4)
edges = cv2.Canny (blur, 50, 200)
edges = cv2.Canny (blur, 50, 200)
edges = cv2.Canny (blur, 50, 200)
result = 255 - edges
cv2.imshow( "Canny (oq-gora)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
