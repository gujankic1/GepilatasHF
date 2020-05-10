'''
Author: Gulyás János
A jelenleg legtöbb funkcióval bíró, végleges script. A dlib előre betanított arcdetektáló és arci jellemző felismerő modelljeit használom a webkamera valós idejű videó-feedjén.
Az így kinyert arci jellemzőkből számolom és figyelem a következőket:
  * Pislogások száma
  * Az arc dőlésszöge (vízszinteshez képest)
  * Mosolygás, meglepődöttség, kiváncsiság, mint mimika detektálása
'''


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

# Szem normált csukottságának számítása
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

# Szemöldök és a szem alsó részének normált távolsága
def curious(eye,eyebrow):
    dist_1 = dist.euclidean(eye[5],eyebrow[2])

    normal = dist.euclidean(eye[0],eye[3])

    curious_lvl = dist_1/normal
    return curious_lvl

# Száj normált nyitottságának számítása
def surprised(lip):
    dist_1 = dist.euclidean(lip[2], lip[10])
    dist_2 = dist.euclidean(lip[4], lip[8])

    normal = dist.euclidean(lip[0], lip[6])

    surprise_lvl = (dist_1 + dist_2) / (2 * normal)
    return surprise_lvl

# Száj normált szélességének számítása
def smiling(lip,eye):
    dist_1 = dist.euclidean(lip[0], lip[6])

    norm_1 = dist.euclidean(eye[0], eye[3])

    smile_lvl = dist_1 / norm_1
    return smile_lvl

# Később felhasznált arc elemek mátrixainak inicializálása
eye_l = np.zeros((6, 2), dtype=int)
eye_r = np.zeros((6, 2), dtype=int)

eyebrow_l = np.zeros((5, 2), dtype=int)
eyebrow_r = np.zeros((5, 2), dtype=int)

outer_lip = np.zeros((12, 2), dtype=int)

# Határértékek a különböző mimikák felismeréséhez
eye_threshold = 0.26
eye_frames = 5

curiosity_threshold=1.3

surprise_threshold=0.6

smile_threshold=2.4

# Változók a pislogások számlálásához
count = 0
blinks = 0

# Emoji-k beolvasása és lekicsinyítése
smile_face = cv2.imread('smile.png')
smile_res = imutils.resize(smile_face, width=50)

curious_face = cv2.imread('curious.png')
curious_res = imutils.resize(curious_face, width=50)

surprised_face = cv2.imread('surprised.png')
surprised_res = imutils.resize(surprised_face, width=50)

offset = np.array((10,10))

face_detector = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor("/home/gujankic/PycharmProjects/GepilatasHF/gepilatas_hf/shape_predictor_68_face_landmarks.dat")

#Indítom a webcam live feed-jét

video_capture = cv2.VideoCapture(0)

while True:
    # Képkockánként olvassuk a képeket
    ret, frame = video_capture.read()
    img_res = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY) # Fekete fehérré konvertálás

    # Használjuk a dlib frontal face detectorát
    faces = face_detector(gray,0)

    # Végigmegyünk az összes talált arcon

    for faces in faces:
        # Meghatározzuk az arci jellemzőket (facial landmarks), a dlib shape predictor-ával, majd ezeket numpy tömbökbe rendezzük
        shape = landmarks(gray, faces)
        shape = shape_to_np(shape)

        # Kimentem a mimika felismeréshez felhasznált arc-elemek koordinátáit
        for j in range(0,6):
            eye_l[j]=shape[j+42]
            eye_r[j]=shape[j+36]

        for k in range(0,5):
            eyebrow_l[k]=shape[k+22]
            eyebrow_r[k]=shape[k+16]

        for p in range(0,12):
            outer_lip[p]=shape[p+48]

        # Az arc horizontális tengelyének szöge
        angle=(1.57-np.arctan(np.abs(shape[36][0]-shape[46][0])/np.abs(shape[36][1]-shape[46][1])))*180/3.14

        # Kiszámítom a mimika felismeréshez szükséges normált távolságokat
        closure = eye_closure_level(eye_l,eye_r)
        curiosity = curious(eye_r,eyebrow_r)
        surprise = surprised(outer_lip)
        smile = smiling(outer_lip,eye_r)

        # Pislogás számlálása, csak akkor számít annak, ha egy előre megadott frame számon keresztül a határértéken belül vagyunk
        if closure<eye_threshold:
            count += 1
        else:
            if count >= eye_frames:
                blinks += 1

            count=0

        # Kíváncsiság (szemöldök felhúzása) felismerése
        if (curiosity>curiosity_threshold)&(surprise < surprise_threshold):

            cv2.putText(img_res, "Curious", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            img_res[offset[0]:offset[0]+curious_res.shape[0],offset[1]:offset[1]+curious_res.shape[1]] = curious_res

        # Meglepődöttség (száj kitátása) felismerése
        if surprise > surprise_threshold:
            cv2.putText(img_res, "Surprised", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            img_res[offset[0]:offset[0] + surprised_res.shape[0], offset[1]:offset[1] + surprised_res.shape[1]] = surprised_res

        # Mosolygás (száj normált hosszának) felismerése
        if smile > smile_threshold:
            cv2.putText(img_res, "Smiling", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            img_res[offset[0]:offset[0] + smile_res.shape[0],offset[1]:offset[1] + smile_res.shape[1]] = smile_res
        '''
        for (x, y) in shape:
            cv2.circle(img_res, (x, y), 1, (0, 0, 255), -1) # A detektált arci jellemzők kirajzolása
        '''
        cv2.putText(img_res, "Angle: {:.2f}".format(angle), (250, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img_res, "Blinks: {:.2f}".format(blinks), (250, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #cv2.putText(img_res, "C: {:.2f}".format(curiosity), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # A 'q' billentyű lenyomásáig közvetítjük a kapott eredményt
    cv2.imshow('Video', img_res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()