import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera ochilmadi")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

    edges = cv2.Canny(gray, 50, 70)

    result = 255 - edges

    cv2.imshow("Canny (oq-gora) - Webcam", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
