# Voiture Autonome

# Importation des bibliothèques
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importation des packages Kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importation de l'objet Dqn depuis notre IA dans ia.py
from ai import Dqn

# Ajout de cette ligne si on ne veut pas que le clic droit place un point rouge
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from timeit import default_timer
start_time = default_timer()
duration_time = default_timer - start_time()


# Définition de last_x et last_y, utilisés pour garder en mémoire le dernier point tracé lorsque l'on dessine le sable sur la carte
last_x = 0
last_y = 0
n_points = 0  # le nombre total de points dans le dernier tracé
length = 0  # la longueur du dernier tracé

# Récupération de notre IA, que l'on appelle "brain", et qui contient notre réseau de neurones représentant notre fonction Q
brain = Dqn(5, 3, 0.9)  # 5 capteurs, 3 actions, gamma = 0.9
# action = 0 => pas de rotation, action = 1 => rotation de 20 degrés, action = 2 => rotation de -20 degrés
action2rotation = [0, 20, -20]
last_reward = 0  # initialisation de la dernière récompense 
# initialisation de la courbe du score moyen (fenêtre glissante des récompenses) en fonction du temps
scores = []

# Initialisation de la carte
first_update = True  # astuce pour initialiser la carte une seule fois


def init():
    # sand est un tableau contenant autant de cellules que notre interface graphique contient de pixels.
    # Chaque cellule vaut 1 s'il y a du sable, 0 sinon.
    global sand
    # coordonnée x de la cible (où la voiture doit aller : l'aéroport ou le centre-ville)
    global goal_x
    # coordonnée y de la cible (où la voiture doit aller : l'aéroport ou le centre-ville)
    global goal_y
    # initialisation du tableau sand rempli de zéros
    sand = np.zeros((longueur, largeur))
    # la cible à atteindre est en haut à gauche de la carte (x=20 et pas 0 car la voiture reçoit une mauvaise récompense si elle touche le mur)
    goal_x = 20
    # la cible à atteindre est en haut à gauche de la carte (coordonnée y)
    goal_y = largeur - 20
    first_update = False  # astuce pour initialiser la carte une seule fois


# Initialisation de la dernière distance
last_distance = 0


# Création de la classe Car (pour comprendre "NumericProperty" et "ReferenceListProperty", voir les tutoriels Kivy : https://kivy.org/docs/tutorials/pong.html)
class Car(Widget):

    # initialisation de l'angle de la voiture (angle entre l'axe x de la carte et l'axe de la voiture)
    angle = NumericProperty(0)
    # initialisation de la dernière rotation de la voiture (après avoir joué l'action : 0°, 20° ou -20°)
    rotation = NumericProperty(0)
    # initialisation de la composante x du vecteur vitesse
    velocity_x = NumericProperty(0)
    # initialisation de la composante y du vecteur vitesse
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)  # vecteur vitesse
    # initialisation de la coordonnée x du premier capteur (qui regarde devant)
    sensor1_x = NumericProperty(0)
    # initialisation de la coordonnée y du premier capteur (qui regarde devant)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)  # vecteur capteur 1
    # initialisation de la coordonnée x du second capteur (qui regarde 30° à gauche)
    sensor2_x = NumericProperty(0)
    # initialisation de la coordonnée y du second capteur (qui regarde 30° à gauche)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)  # vecteur capteur 2
    # initialisation de la coordonnée x du troisième capteur (qui regarde 30° à droite)
    sensor3_x = NumericProperty(0)
    # initialisation de la coordonnée y du troisième capteur (qui regarde 30° à droite)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)  # vecteur capteur 3
    # initialisation du signal reçu par le capteur 1
    signal1 = NumericProperty(0)
    # initialisation du signal reçu par le capteur 2
    signal2 = NumericProperty(0)
    # initialisation du signal reçu par le capteur 3
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # mise à jour de la position de la voiture en fonction de sa dernière position et vitesse
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation  # récupération de la rotation appliquée
        self.angle = self.angle + self.rotation  # mise à jour de l'angle
        # mise à jour de la position du capteur 1
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        # mise à jour de la position du capteur 2
        self.sensor2 = Vector(30, 0).rotate((self.angle+30) % 360) + self.pos
        # mise à jour de la position du capteur 3
        self.sensor3 = Vector(30, 0).rotate((self.angle-30) % 360) + self.pos
        # récupération du signal reçu par le capteur 1 (densité de sable autour du capteur 1)
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10,
                                      int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        # récupération du signal reçu par le capteur 2 (densité de sable autour du capteur 2)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10,
                                      int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        # récupération du signal reçu par le capteur 3 (densité de sable autour du capteur 3)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10,
                                      int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        # si le capteur 1 est en dehors de la carte (voiture face à un bord)
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
            self.signal1 = 1.  # capteur 1 détecte du sable plein
        # si le capteur 2 est en dehors de la carte
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
            self.signal2 = 1.
        # si le capteur 3 est en dehors de la carte
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
            self.signal3 = 1.


class Ball1(Widget):  # capteur 1 (voir tutoriels Kivy : https://kivy.org/docs/tutorials/pong.html)
    pass


class Ball2(Widget):  # capteur 2
    pass


class Ball3(Widget):  # capteur 3
    pass


# Création de la classe Game (pour comprendre "ObjectProperty", voir tutoriels Kivy : https://kivy.org/docs/tutorials/pong.html)
class Game(Widget):

    car = ObjectProperty(None)  # récupération de l'objet voiture depuis le fichier kivy
    ball1 = ObjectProperty(None)  # récupération du capteur 1
    ball2 = ObjectProperty(None)  # récupération du capteur 2
    ball3 = ObjectProperty(None)  # récupération du capteur 3

    def serve_car(self):  # lancement de la voiture lorsque l'application démarre
        self.car.center = self.center  # la voiture démarre au centre
        self.car.velocity = Vector(6, 0)  # vitesse initiale horizontale vers la droite

    def update(self, dt):  # grande fonction de mise à jour appelée à chaque frame

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global start_time
        global duration_time

        longueur = self.width  # largeur de la carte
        largeur = self.height  # hauteur de la carte
        if first_update:
            init()

        xx = goal_x - self.car.x  # différence en x
        yy = goal_y - self.car.y  # différence en y
        orientation = Vector(*self.car.velocity).angle(
            (xx, yy)) / 180.  # orientation de la voiture par rapport à la cible
        # vecteur d'état : 3 signaux + orientation + -orientation
        last_signal = [self.car.signal1, self.car.signal2,
                       self.car.signal3, orientation, -orientation, duration_time]
        # action jouée par l'IA
        action = brain.update(last_reward, last_signal)
        # ajout du score
        scores.append(brain.score())
        # conversion action → rotation
        rotation = action2rotation[action]
        # déplacement de la voiture
        self.car.move(rotation)
        # nouvelle distance à la cible
        distance = np.sqrt((self.car.x - goal_x)**2 +
                           (self.car.y - goal_y)**2)
        # mise à jour des positions des capteurs
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:  # voiture sur le sable
            self.car.velocity = Vector(1, 0).rotate(
                self.car.angle)  # ralentie
            last_reward = -1  # mauvaise récompense
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2  # mauvaise récompense légère
            if distance < last_distance:  # si elle se rapproche de la cible
                last_reward = 0.1  # petite récompense positive

        # gestion des collisions avec les bords
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width-10:
            self.car.x = self.width-10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height-10:
            self.car.y = self.height-10
            last_reward = -1

        # inversion de la cible lorsqu’elle est atteinte
        if distance < 100:
            last_reward = 1.5
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
            start_time = default_timer()

        duration_time = default_timer() - start_time
        if duration_time > 10:
            last_reward = -1
        else:
            last_reward = 0.5
            
        last_distance = distance


# Dessin pour l'interface graphique (voir tutoriels : https://kivy.org/docs/tutorials/firstwidget.html)
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):  # placement du sable lors d'un clic gauche
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    # placement du sable lors du déplacement de la souris clic gauche enfoncé
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 +
                                  (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10,
                 int(touch.y) - 10: int(touch.y) + 10] = 1
            last_x = x
            last_y = y


# Interface API et boutons (voir tutoriels Kivy : https://kivy.org/docs/tutorials/pong.html)
class CarApp(App):

    def build(self):  # construction de l'app
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2*parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):  # bouton clear
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):  # bouton save
        print("sauvegarde du cerveau…")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):  # bouton load
        print("chargement du dernier cerveau sauvegardé…")
        brain.load()


# Lancement de l'app
if __name__ == '__main__':
    CarApp().run()
 