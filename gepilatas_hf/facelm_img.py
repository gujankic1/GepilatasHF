import numpy as np
import argparse
import dlib
import cv2

faceCascade = cv2.CascadeClassifier('/home/gujankic/PycharmProjects/gepilatas_hf/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Mátrixba írjuk át a dlib landmark detector shape visszatérési objektumát
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype) # Inicializáljuk a 68 pont mátrixát
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y) # Beírjuk az x-y koordinátákat
	return coords

img = cv2.imread('haarcasc_img.jpg') # Kép beolvasása
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100)) # Arcok detektálása. Adott képre a legjobb paraméterekkel. Rectangle-el tér vissza, ha talál.

predictor = dlib.shape_predictor(args["shape_predictor"])