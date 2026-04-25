# ================================
# Image Preprocessing
# ================================

# Importation des bibliothèques nécessaires
import numpy as np
import cv2
from gymnasium.core import ObservationWrapper
from gymnasium.spaces.box import Box


class PreprocessImage(ObservationWrapper):
    """
    Wrapper Gym permettant de prétraiter les observations visuelles :
    - redimensionnement
    - passage en niveaux de gris (optionnel)
    - normalisation
    - changement de format des dimensions
    """

    def __init__(self, env, height=124, width=124, grayscale=True, crop=lambda img: img):
        # Initialisation du wrapper Gym
        super(PreprocessImage, self).__init__(env)

        # Taille finale de l'image (hauteur, largeur)
        self.img_size = (height, width)

        # Indique si l'image doit être convertie en niveaux de gris
        self.grayscale = grayscale

        # Fonction de recadrage (crop) appliquée à l'image
        self.crop = crop

        # Nombre de canaux de couleur (1 = gris, 3 = RGB)
        n_colors = 1 if self.grayscale else 3

        # Définition de l'espace des observations après prétraitement
        # Valeurs normalisées entre 0 et 1
        # Format : (canaux, hauteur, largeur)
        self.observation_space = Box(
            0.0, 1.0, [n_colors, height, width]
        )

    def observation(self, img):
        """
        Applique le prétraitement à une image brute issue de l'environnement
        """

        # Application du recadrage personnalisé
        img = self.crop(img)

        # Redimensionnement de l'image à la taille souhaitée
        img = cv2.resize(img, self.img_size)

        # Conversion en niveaux de gris si nécessaire
        if self.grayscale:
            # Moyenne des canaux RGB → image en 1 canal
            img = img.mean(-1, keepdims=True)

        # Changement de l'ordre des dimensions :
        # (hauteur, largeur, canaux) → (canaux, hauteur, largeur)
        img = np.transpose(img, (2, 0, 1))

        # Conversion en float32 et normalisation des pixels entre 0 et 1
        img = img.astype('float32') / 255.0

        return img
