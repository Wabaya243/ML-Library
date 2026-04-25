# Voiture autonome

# Importation des bibliothèques
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt


# Importation des modules Kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importation de l'objet Dqn depuis ai.py (notre IA)
from ai import Dqn

# Ligne ajoutée pour éviter que le clic droit dépose un point rouge
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Variables pour mémoriser le dernier point dessiné (sable)
last_x = 0
last_y = 0
n_points = 0
length = 0

# Initialisation de l'IA ("brain") contenant le réseau de neurones représentant la Q-fonction
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 20, -20]
last_reward = 0
scores = []

# Initialisation de la carte
first_update = True


def init():
    # Initialise la carte et la position de l’objectif
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False


# Initialisation de la dernière distance
last_distance = 0


# Création de la classe Car (la voiture)
class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # Déplacement de la voiture + mise à jour des capteurs et signaux
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30) % 360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(
            self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(
            self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(
            self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

        # Si un capteur touche les bords : signal maximal (danger)
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
            self.signal3 = 1.


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Création de la classe Game (le jeu principal)
class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        # Position initiale de la voiture
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        # Mise à jour du jeu : IA, récompense, distance, limites, objectif
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        

        longueur = self.width
        largeur = self.height

        if first_update:
            init()

        # Calcul de l’orientation et envoi des signaux à l’IA
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.
        last_signal = [self.car.signal1, self.car.signal2,
                       self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)

        # Calcul de la distance objectif
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Récompenses selon si on touche le sable ou non
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        # Gestion des collisions avec les bords
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        # Si la voiture approche l’objectif → déplacer l’objectif
        if distance < 100:
            last_reward = 1.5
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
    
        last_distance = distance


# Outil de dessin du sable
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        # Quand l’utilisateur dessine du sable
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

    def on_touch_move(self, touch):
        # Mise à jour du tracé et densité du sable
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10,
                 int(touch.y) - 10: int(touch.y) + 10] = 1
            last_x = x
            last_y = y


# Ajout des boutons clear / save / load
class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        # Effacer la carte
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        # Sauvegarde du cerveau de l’IA et affichage des scores
        print("sauvegarde du cerveau...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        # Chargement de la dernière sauvegarde IA
        print("chargement du dernier cerveau sauvegardé...")
        brain.load()


# Exécution de l'application
if __name__ == '__main__':
    CarApp().run()
