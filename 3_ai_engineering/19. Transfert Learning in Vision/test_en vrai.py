######## pour cifar 10 ###########
import tensorflow as tf
import numpy as np
import os
import shutil
from PIL import Image

# Dossier de sortie
base_dir = "dataset_cifar10"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Classes CIFAR-10
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Nettoyer si déjà présent
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

# Charger CIFAR-10 depuis Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Création des dossiers
for split_dir in [train_dir, val_dir]:
    for class_name in class_names:
        os.makedirs(os.path.join(split_dir, class_name))

# Sauvegarder les images d'entraînement
for idx, (img, label) in enumerate(zip(x_train, y_train)):
    class_name = class_names[int(label)]
    img_path = os.path.join(train_dir, class_name, f"{idx}.png")
    Image.fromarray(img).resize((244, 244)).save(img_path)

# Sauvegarder les images de validation
for idx, (img, label) in enumerate(zip(x_test, y_test)):
    class_name = class_names[int(label)]
    img_path = os.path.join(val_dir, class_name, f"{idx}.png")
    Image.fromarray(img).resize((244, 244)).save(img_path)

print("✅ Dataset CIFAR-10 téléchargé et structuré dans", base_dir)

