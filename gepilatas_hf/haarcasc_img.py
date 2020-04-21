import numpy as np
import cv2

# Betöltjük az OpenCV által biztosított kaszkádokat, egyet az arcra (előről) és egyet a szemekre

faceCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml')

img = cv2.imread('haarcasc_img.jpg') # Kép beolvasása
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100)) # Arcok detektálása. Adott képre a legjobb paraméterekkel. Rectangle-el tér vissza, ha talál.

# Körülrajzoljuk az arcokat, majd az arcon belül keressük a szemeket és azokat is körberajzoljuk.

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()