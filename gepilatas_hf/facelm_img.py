import numpy as np
import argparse
import dlib
import cv2
import imutils

faceCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Mátrixba írjuk át a dlib landmark detector shape visszatérési objektumát
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype) # Inicializáljuk a 68 pont mátrixát
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y) # Beírjuk az x-y koordinátákat
	return coords


def bb_to_rect(x,y,w,h):
	tmp_right=x+w
	tmp_bottom=h+y
	tmp_rect=dlib.rectangle(x,y,tmp_right,tmp_bottom)
	tmp_rect
	return(tmp_rect)



img = cv2.imread('haarcasc_img.jpg') # Kép beolvasása
img_res = imutils.resize(img, width=500)
gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100)) # Arcok detektálása. Adott képre a legjobb paraméterekkel. Rectangle-el tér vissza, ha talál.

landmarks = dlib.shape_predictor("/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/shape_predictor_68_face_landmarks.dat")

# Végigmegyünk az összes talált arcon

for (x,y,w,h) in faces:
	# Meghatározzuk az arci jellemzőket (facial landmarks), a dlib shape predictor-ával, majd ezeket numpy tömbökbe rendezzük
	shape = landmarks(gray, bb_to_rect(x,y,w,h))
	shape = shape_to_np(shape)

	# A dlib rectangle osztályát a cv2 által preferált bounding box-á alakítjuk
	cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# Kirajzoljuk az arci jellemzőket
	for (x, y) in shape:
		cv2.circle(img_res, (x, y), 1, (0, 0, 255), -1)

# Kidobjuk a képet
cv2.imshow("Output", img_res)
cv2.waitKey(0)