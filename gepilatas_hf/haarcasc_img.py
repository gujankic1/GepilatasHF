'''
Author: Gulyás János
Az OpenCV Haar-kaszkádokat használó arcdetektálójának kipróbálása egy beolvasott képen. Ha a képen több arc is található, mindegyiket körülrajzolja.
Ezen felül az ugyancsak Haar-kaszkádokat használó szemdetektálót is használjuk, minden arcon belül megkeresi és kirajzolja a szemeket is.
'''


import numpy as np
import cv2

# Betöltjük az OpenCV által biztosított kaszkádokat, egyet az arcra (előről) és egyet a szemekre

faceCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml')

img_1 = cv2.imread('haarcasc_img.jpg') # Kép beolvasása
img_2 = cv2.imread('ppl.jpg')

gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

faces_1 = faceCascade.detectMultiScale(gray_1, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100)) # Arcok detektálása. Adott képre a legjobb paraméterekkel. Rectangle-el tér vissza, ha talál.
faces_2 = faceCascade.detectMultiScale(gray_2, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30)) # Kisebb arcok, kisebb minSize értékkel működik

# Körülrajzoljuk az arcokat, majd az arcon belül keressük a szemeket és azokat is körberajzoljuk.

for (x,y,w,h) in faces_1:
    img = cv2.rectangle(img_1,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_1[y:y+h, x:x+w]
    roi_color = img_1[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

for (x,y,w,h) in faces_2:
    img = cv2.rectangle(img_2,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_2[y:y+h, x:x+w]
    roi_color = img_2[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.01, minNeighbors=3, minSize=(5, 5))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('Output_1',img_1)
cv2.imshow('Output_2',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()