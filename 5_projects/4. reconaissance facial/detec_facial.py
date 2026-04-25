import cv2


# Charger le classificateur en cascade pré-entraîné pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces_from_image(imagePath):
    #charger l'image
    image = cv2.imread("face.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #detecter la face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    #dessiner le rectangle autour de l'image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x + w, y + h), (0, 0, 255), 2)
        #afficher l'image
    cv2.imshow("Detection de visage", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows(0)

def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detecter les visagee
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        #dessiner les rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 2)
        
        #afficher les resultat
        cv2.imshow("Detection des visages en temps-reel", frame)

        #appuyer sur q pour quitté
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #liberer la memoire
    cap.release()
    cv2.destroyAllWindows()


#lancé les programes
print("choisisez une opion detecter a partir d'une image ou a partir du webcam")
choice = input("Entrez 1 ou 2 : ")

if choice == "1":
    image_path = input("Entrez le chemin de l'image :")
    detect_faces_from_image(image_path)

elif choice == "2":
    detect_from_webcam()

else :
    print("choix invalide. sortie")





#charger l'image
image = cv2.imread("face.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detecter la face
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

#dessiner le rectangle autour de l'image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x + w, y + h), (0, 0, 255), 2)

#afficher l'image
cv2.imshow("Detection de visage", image)
cv2.waitKey(0)
cv2.destroyAllWindows(0)

#initialisé les webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecter les visagee
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    #dessiner les rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 2)
    
    #afficher les resultat
    cv2.imshow("Detection des visages en temps-reel", frame)

    #appuyer sur q pour quitté
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#liberer la memoire
cap.release()
cv2.destroyAllWindows()

