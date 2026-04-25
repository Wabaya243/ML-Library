import cv2 as cv

#charger les classificateurs de visage
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

#charge les images
img = cv.imread('visage.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#on execute la detection des visage 
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# recuperation des coordoné des chaque visage 
    
#affiches les visages
for face in faces:
    x, y, w, h = face

    #dessinner le rectangle sur l'image
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#execution de la detection des yeux
eyes = eyes_cascade.detectMultiScale(gray, 1.1, 3)

for eye in eyes:
    ex, ey, ew, eh = eye
    cv.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


#affiche l'image principale
cv.imshow('Visage', img)
cv.WaitKey(0)
cv.DestroyAllWindows()


######### Partie 2


#charge les images
img = cv.imread('visage.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#on execute la detection des visage 
faces = face_cascade.detectMultiScale(gray, 1.3, 6)

# recuperation des coordoné des chaque visage 
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

#extraction de 2 visage 
face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

#redimensioner les visage
face1 = cv.resize(face1, (w2, h2))
face2 = cv.resize(face2, (w1, h1))

#remplacé le visage 1 par le visage 2
img[y1:y1+h1, x1:x1+w1] = face2

#remplacé le visage 2 par le visage 1
img[y2:y2+h2, x2:x2+w2] = face1

#afficher l'image
cv.imshow('echange', img)
cv.WaitKey(0)
cv.DestroyAllWindows()