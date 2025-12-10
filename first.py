import cv2

img = cv2.imread('img_2.png')
cv2.imshow("Original", img)

# Non-Local Means (rangli tasvir)
nlm = cv2.fastNlMeansDenoisingColored(
    img,
    None,
    h=10,           # kuchlilik koeffitsiyenti
    hColor=10,
    templateWindowSize=7,
    searchWindowSize=21
)

cv2.imshow("Non-Local Means Denoise", nlm)

cv2.waitKey(0)
cv2.destroyAllWindows()
