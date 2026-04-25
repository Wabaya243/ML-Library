# On impotes les librairies

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def center_and_resize(img, size=28):
    # Binarisation
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    # Bounding box autour du plus grand contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Extraire le chiffre
    digit = img[y:y+h, x:x+w]

    # Redimensionner à 20x20 (comme dans MNIST)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Centrer dans un canvas 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    return canvas




#Chargement des données
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# On reshape le data
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) 

# On verefie le shape apres le reshaping
print(x_test.shape)
print(x_train.shape)

#normalisation le pixel des valeurs
x_train = x_train/255
x_test = x_test/255



#je rajoute les images en local

# Chemins et labels
paths = [
    ("mon_chiffre_1.png", 1), ("mon_chiffre_2.png", 2), ("mon_chiffre_3.png", 3),
    ("mon_chiffre_4.png", 4), ("mon_chiffre_5.png", 5), ("mon_chiffre_6.jpg", 6),
    ("mon_chiffre_7.png", 7), ("mon_chiffre_8.png", 8), ("mon_chiffre_9.png", 9)
]

X_custom = []
y_custom = []

for path, label in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    img = center_and_resize(img)
    img = 255 - img  # Inversion
    img = img.astype('float32') / 255.0
    img = img.reshape(28, 28, 1)
    
    X_custom.append(img)
    y_custom.append(label)

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)


# On cree 30 variation de la meme image pour augmenter la precision de l'entrainement
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

X_aug = []
y_aug = []

for img, label in zip(X_custom, y_custom):
    img = img.reshape(1, 28, 28, 1)
    count = 0
    for batch in datagen.flow(img, batch_size=1):
        X_aug.append(batch[0])
        y_aug.append(label)
        count += 1
        if count >= 30:  # Crée 30 variations par image
            break

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)



# je combine les mnist et mes images

x_train_combined = np.concatenate([x_train, X_custom], axis=0)
y_train_combined = np.concatenate([y_train, y_custom], axis=0)

x_train_combined = np.concatenate([x_train_combined, X_aug], axis=0)
y_train_combined = np.concatenate([y_train_combined, y_aug], axis=0)



#je melange des manieres aleatoire parce que c'est conseillé
from sklearn.utils import shuffle
x_train_combined, y_train_combined = shuffle(x_train_combined, y_train_combined, random_state=42)



#creation du modele
model = Sequential()

# L'ajout du convolution layer
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))



# Ajout de la phase de flattening
model.add(Flatten())

# Ajout de la phase de full connection
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# Ajout de la phase de la sortie
model.add(Dense(10, activation='softmax'))

# Compilation du modele
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#pour sauvegarder automatiquement pendant l'entrainement
checkpoint = ModelCheckpoint('Save/temp_mnist_cnn_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Entrainement du modele
history = model.fit(x_train_combined, y_train_combined, epochs=10, batch_size=128, validation_split=0.2, callbacks=[checkpoint])


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title("Évolution de la précision")
plt.show()




# Chargement du meilleur modèle
from sklearn.metrics import roc_auc_score
import os

def calcul_auc(model, x_test, y_test):
    y_proba = model.predict(x_test)
    return roc_auc_score(y_test, y_proba, multi_class='ovr')

def charger_et_comparer_models(nouveau_path, ancien_path, x_test, y_test):
    if not os.path.exists(nouveau_path):
        raise FileNotFoundError(" Aucun modèle temporaire trouvé après l'entraînement.")

    print("\n📥 Chargement du modèle nouvellement entraîné...")
    nouveau_modele = load_model(nouveau_path)
    auc_nouveau = calcul_auc(nouveau_modele, x_test, y_test)

    if os.path.exists(ancien_path):
        print(" Chargement de l'ancien modèle...")
        ancien_modele = load_model(ancien_path)
        auc_ancien = calcul_auc(ancien_modele, x_test, y_test)

        print(f"\n Ancien AUC : {auc_ancien:.3f}")
        print(f" Nouveau AUC : {auc_nouveau:.3f}")

        if auc_nouveau > auc_ancien:
            print(" Nouveau modèle meilleur → Remplacement effectué.")
            nouveau_modele.save(ancien_path)
            return nouveau_modele
        else:
            print(" Ancien modèle conservé (meilleur AUC).")
            return ancien_modele
    else:
        print(" Aucun modèle précédent. Le modèle actuel devient le meilleur.")
        nouveau_modele.save(ancien_path)
        return nouveau_modele

meilleur_modele = charger_et_comparer_models(
    "Save/temp_mnist_cnn_model.keras",
    "Save/mnist_cnn_model.keras",
    x_test,
    y_test
)

# evaluation du modele
meilleur_modele.evaluate(x_test,y_test)




## la partie test en local


def load_and_prepare_image(filepath):
    """
    Charge une image, la centre, la redimensionne, l'inverse, la normalise
    et la reshape pour la prédiction.
    
    Args:
        filepath (str): chemin du fichier image
    
    Returns:
        np.ndarray: image prête à être prédite par le modèle (1, 28, 28, 1)
    """
    # Lire en niveaux de gris
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Image non trouvée ou illisible : {filepath}")
    
    # Centrage et redimensionnement (doit être définie avant)
    img = center_and_resize(img)
    
    # Inversion des couleurs
    img = 255 - img
    
    # Normalisation [0, 1]
    img = img.astype('float32') / 255.0
    
    # Reshape pour correspondre au modèle
    img = img.reshape(1, 28, 28, 1)
    
    return img




# Appliquer cette fonction à chaque image perso avant inversion / normalisation
img1 = load_and_prepare_image("mon_chiffre_1.png")
img2 = load_and_prepare_image("mon_chiffre_2.png")
img3 = load_and_prepare_image("mon_chiffre_3.png")
img4 = load_and_prepare_image("mon_chiffre_4.png")
img5 = load_and_prepare_image("mon_chiffre_5.png")
img6 = load_and_prepare_image("mon_chiffre_6.png")
img7 = load_and_prepare_image("mon_chiffre_7.png")
img8 = load_and_prepare_image("mon_chiffre_8.png")
img9 = load_and_prepare_image("mon_chiffre_9.png")



import matplotlib.pyplot as plt

for i, (img, label) in enumerate(zip([img1, img2, img3, img4, img5, img6, img7, img8, img9], range(1, 10))):
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Image perso {label}")
    plt.axis('off')
    plt.show()

    pred = meilleur_modele.predict(img)
    print(f"Class prédite pour {label} :", np.argmax(pred))





## pour entrainer avec les images en local



# Crée X_custom et y_custom pour fine-tuning
X_custom = np.concatenate([img1, img2, img3, img4, img5, img6, img7, img8, img9], axis=0)
y_custom = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Réentraîner quelques epochs (évite le surapprentissage)
meilleur_modele.fit(X_custom, y_custom, epochs=5, batch_size=2, verbose=1)
meilleur_modele.save('Save/mnist_cnn_model.keras')