import cv2
import numpy as np
import math

# Rasmni yuklash
img = cv2.imread("img.png")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eyes_cascade.detectMultiScale(roi_gray)

    if len(eyes) >= 2:

        # faqat 2 ta ko'zni olamiz
        eye1 = eyes[0]
        eye2 = eyes[1]

        # ko'z markazlari
        x1 = x + eye1[0] + eye1[2] // 2
        y1 = y + eye1[1] + eye1[3] // 2

        x2 = x + eye2[0] + eye2[2] // 2
        y2 = y + eye2[1] + eye2[3] // 2

        # chap va o‘ng bo‘lishi uchun tartiblash
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        # Burchakni hisoblash
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))

        # teskari burilishlarga yo‘l qo‘ymaslik
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        # Rasmni aylantirish
        (h_img, w_img) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w_img, h_img))

        # saqlash va ko‘rsatish
        cv2.imwrite("img12.png", rotated)
        cv2.imshow("Natija", rotated)
        cv2.waitKey(0)

# asl rasmni ko‘rsatish
cv2.imshow("Asl rasm", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
