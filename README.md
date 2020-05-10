# Arc és mimika felismerés  
Gépi látás fakultatív házi feladat. A projekt során az OpenCV és a dlib által biztosított arcfelismerőket és arci jellemző (facial landmark) felismerőket próbáltam ki és használtam fel egy arcot követő és alap mimikákat felismerő script megírásához.

## Általános információ
Eredetileg egy a felhasználót valós időben követő, *virtuális jólét figyelő*-szerű alkalmazás fejlesztése volt a cél. Ehhez először az OpenCV cv2 könyvtárában található, Haar-kaszkádos arcfelismerőt próbáltam ki, mind képeken, mind a webkamera live videó feed-jén. Azonban ez az arcfelismerő modul nem annyira robosztus, nem viseli túl jól az arc mozgatását, döntését, frontális arcdetektálásra való igazán. 

Egy robosztusabb megoldást biztosít a dlib frontális arcfelismerője. Ez jobban tűri a különböző szögű arcokat és általánosságban véve gyorsabb is, így alkalmasabb a valós idejű arcdetektálásra. Ugyancsak a dlib könyvtár biztosít egy arci jellemzőket felismerő algoritmust. 68 nevezetes arci jellemzőt keres meg a már detektált arc területén belül. A jelenlegi alkalmazás verzió ezt a két technológiát kombinálva figyeli a felhasználó arcát valós időben a webkamerán keresztül.

## Technológiák
A felhasznált technológiák és verziószámaik

* Pycharm Community 2020.1.1
* Python 3.5

## Setup  
Git-ről a teljes GepilatasHF repo-t klónozva, majd Pycharm-ban (vagy más, szimpatikus python IDE-ben) projektként megnyitva, használható az összes script. Virtual environment-et használtam a lehető legjobb portolhatóság érdekében, azaz a `venv` mappában megtalálható minden szükséges könyvtár a projekthez. A kódokon belül az importált file-ok helyét manuálisan kell átírni jelenleg.

## Funkciók

* `haarcasc_img.py` - Az OpenCV Haar-kaszkádokat használó arcdetektálójának kipróbálása egy beolvasott képen. Ha a képen több arc is található, mindegyiket körülrajzolja. Ezen felül az ugyancsak Haar-kaszkádokat használó szemdetektálót is használjuk, minden arcon belül megkeresi és kirajzolja a szemeket is. 
* `haarcascade_video.py` - Az OpenCV Haar-kaszkádokat használó arcdetektálójának kipróbálása a webkamera valós idejű videó-feedjén. Ha a videón több arc is található, mindegyiket körülrajzolja. Ezen felül az ugyancsak Haar-kaszkádokat használó szemdetektálót is használjuk, minden arcon belül megkeresi és kirajzolja a szemeket is. 
* `facelm_img.py` - Még ebben a scriptben is a Haar-kaszkádos arcdetektálót használjuk. Az arcok megtalálása után megkeressük és kirajzoljuk a 68 nevezetes arci jellemzőt (facial landmarks) mindegyik arcon a dlib előre betanított modelljével.
* `facelm_vid.py` - A jelenleg legtöbb funkcióval bíró, végleges script. A dlib előre betanított arcdetektáló és arci jellemző felismerő modelljeit használom a webkamera valós idejű videó-feedjén. Az így kinyert arci jellemzőkből számolom és figyelem a következőket:
  * Pislogások száma
  * Az arc dőlésszöge (vízszinteshez képest)
  * Mosolygás, meglepődöttség, kiváncsiság, mint mimika detektálása

## Fejlesztési lehetőségek
* További mimikák/érzelmek felismerése az arci jellemzők alapján
* A tartásról teljesebb kép kinyerése (esetleg Aruco markerek elhelyezésével a vállakon)
* Szemüveges felhasználók jobb kezeléséhez új modellek tanítása

## Kontakt

A projektet készítette:

* Gulyás János (gujankic@gmail.com | @gujankic1)
