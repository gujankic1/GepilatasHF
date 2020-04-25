from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist


# Mátrixba írjuk át a dlib landmark detector shape visszatérési objektumát
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype) # Inicializáljuk a 68 pont mátrixát
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y) # Beírjuk az x-y koordinátákat
	return coords

def eye_closure_level(eye_l, eye_r):

    dist_l_1 = dist.euclidean(eye_l[1],eye_l[5])
    dist_r_1 = dist.euclidean(eye_r[1], eye_r[5])

    dist_l_2 = dist.euclidean(eye_l[2], eye_l[4])
    dist_r_2 = dist.euclidean(eye_r[2], eye_r[4])

    dist_l_3 = dist.euclidean(eye_l[0], eye_l[3])
    dist_r_3 = dist.euclidean(eye_r[0], eye_r[3])

    closure_r = (dist_r_2+dist_r_1)/(2*dist_r_3)
    closure_l = (dist_l_2+dist_l_1)/(2*dist_l_3)

    closure_avg=(closure_r+closure_l)/2
    return closure_avg

def curious(eye,eyebrow):
    dist_1 = dist.euclidean(eye[1],eyebrow[2])
    dist_2 = dist.euclidean(eye[2],eyebrow[3])

    normal = dist.euclidean(eye[0],eye[3])

    curious_lvl = (dist_1+dist_2)/(2*normal)
    return curious_lvl

def surprised(lip):
    dist_1 = dist.euclidean(lip[2], lip[10])
    dist_2 = dist.euclidean(lip[4], lip[8])

    normal = dist.euclidean(lip[0], lip[6])

    surprise_lvl = (dist_1 + dist_2) / (2 * normal)
    return surprise_lvl

eye_l = np.zeros((6, 2), dtype=int)
eye_r = np.zeros((6, 2), dtype=int)

eyebrow_l = np.zeros((5, 2), dtype=int)
eyebrow_r = np.zeros((5, 2), dtype=int)

outer_lip = np.zeros((12, 2), dtype=int)

eye_threshold = 0.26
eye_frames = 4
# initialize the frame counters and the total number of blinks
count = 0
blinks = 0

curiosity_threshold=1.1

surprise_threshold=0.6

face_detector = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor("/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/shape_predictor_68_face_landmarks.dat")

#Indítom a webcam live feed-jét

video_capture = cv2.VideoCapture(0)

while True:
    # Képkockánként olvassuk a képeket
    ret, frame = video_capture.read()
    img_res = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás

    faces = face_detector(gray,0)

    # loop over the face detections

    for faces in faces:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = landmarks(gray, faces)
        shape = shape_to_np(shape)

        for j in range(0,6):
            eye_l[j]=shape[j+42]
            eye_r[j]=shape[j+36]

        for k in range(0,5):
            eyebrow_l[k]=shape[k+22]
            eyebrow_r[k]=shape[k+16]

        for p in range(0,12):
            outer_lip[p]=shape[p+48]

        closure=eye_closure_level(eye_l,eye_r)
        curiosity=curious(eye_r,eyebrow_r)
        surprise=surprised(outer_lip)

        if closure<eye_threshold:
            count += 1
        else:
            if count >= eye_frames:
                blinks += 1

            count=0

        if curiosity>curiosity_threshold:

            cv2.putText(img_res, "Curious", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if surprise > surprise_threshold:
            cv2.putText(img_res, "Surprised", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(img_res, (x, y), 1, (0, 0, 255), -1)

        cv2.putText(img_res, "Closure: {:.2f}".format(closure), (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img_res, "Blinks: {:.2f}".format(blinks), (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #cv2.putText(img_res, "C: {:.2f}".format(surprise), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Video', img_res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()