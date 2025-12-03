import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

img = cv2.imread("img.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    if len(eyes) >= 2:

        # Faqat eng katta ikkita ko‘zni olamiz
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]

        # Ko‘zlar markazlari
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            cx = x + ex + ew//2
            cy = y + ey + eh//2
            eye_centers.append((cx, cy))

        # Ko‘zlarni chap-o‘ng bo‘yicha tartiblash
        eye_centers = sorted(eye_centers, key=lambda p: p[0])

        left_eye = eye_centers[0]
        right_eye = eye_centers[1]

        # Burchakni hisoblash
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))

        # Tasvirni to‘g‘ri aylantirish
        center = (img.shape[1]//2, img.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        cv2.imshow("Normal holdagi tasvir", aligned)

cv2.imshow("Asl tasvir", img)
cv2.waitKey(0)
cv2.destroyAllWindows()