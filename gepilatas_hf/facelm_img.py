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

# loop over the face detections

for (x,y,w,h) in faces:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = landmarks(gray, bb_to_rect(x,y,w,h))
	shape = shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# show the face number
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(img_res, (x, y), 1, (0, 0, 255), -1)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", img_res)
cv2.waitKey(0)