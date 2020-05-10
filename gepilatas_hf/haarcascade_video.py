'''
Author: Gulyás János
Az OpenCV Haar-kaszkádokat használó arcdetektálójának kipróbálása a webkamera valós idejű videó-feedjén. Ha a videón több arc is található, mindegyiket körülrajzolja.
Ezen felül az ugyancsak Haar-kaszkádokat használó szemdetektálót is használjuk, minden arcon belül megkeresi és kirajzolja a szemeket is.
'''

import cv2
import numpy as np
import sys

# Betöltjük az OpenCV által biztosított kaszkádokat, egyet az arcra (előről) és egyet a szemekre

faceCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml')

#Indítom a webcam live feed-jét

video_capture = cv2.VideoCapture(0)


while True:
    # Képkockánként olvassuk a képeket
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás

    # Körülrajzoljuk az arcokat, majd az arcon belül keressük a szemeket és azokat is körberajzoljuk.

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100))
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(roi_gray , scaleFactor=1.15, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Real-time dobom ki a detektált objektumokat
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

