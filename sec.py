import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera ochilmadi")
    exit()

kernel_roberts_x = np.array([[1, 0],
                             [0, -1]], dtype=np.float32)

kernel_roberts_y = np.array([[0, 1],
                             [-1, 0]], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_roberts_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_roberts_y)

    edge = cv2.sqrt(gx * gx + gy * gy)
    edge = cv2.convertScaleAbs(edge)

    result = 255 - edge

    cv2.imshow("Roberts algoritmi (Webcam)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
