import sys, os
import cv2
import numpy as np
from ultralytics import YOLO

# Charger le modèle YOLOv5 (en réalité, YOLOv8 sous le capot, rétro-compatible)
model = YOLO("yolov5s.pt")


def silent_infer(image):
    # bloquer toute sortie dans la console pendant l'inférence
    sys.stdout = open(os.devnull, 'w')
    results = model(image, verbose=False)
    sys.stdout = sys.__stdout__
    return results


def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prédiction
        results = silent_infer(frame)

        # Affichage OpenCV
        annotated = results[0].plot()
        cv2.imshow("Détection d'objet en temps réel", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_objet(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    annotated = results[0].plot()
    cv2.imshow("Résultat", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Lancer le programme
print("Choisissez une option :")
print("1 - Détecter à partir d'une image")
print("2 - Détecter à partir de la webcam")

choice = input("Entrez 1 ou 2 : ")

if choice == "1":
    image_path = input("Entrez le chemin de l'image : ")
    detect_objet(image_path)

elif choice == "2":
    detect_from_webcam()

else:
    print("Choix invalide. Sortie.")
