# ================================
# 📦 Import des librairies
# ================================
from tensorflow.keras.datasets import mnist  # Dataset MNIST
from tensorflow.keras.models import Sequential, load_model  # Pour créer et charger un modèle
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D  # Couches utiles pour CNN
from tensorflow.keras.callbacks import ModelCheckpoint  # Pour sauvegarder le meilleur modèle
import numpy as np
import cv2  # Pour traitement d'image OpenCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Pour l'augmentation de données


# ================================
# 🖼️ Fonction : centrage et redimensionnement des chiffres
# ================================
def center_and_resize(img, size=28):
    # Binarisation de l'image (noir et blanc) avec seuil automatique Otsu
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Détection des contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:  # Si pas de contour → image vide
        return np.zeros((size, size), dtype=np.uint8)

    # Prend le plus grand contour (le chiffre principal)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Découpe le chiffre
    digit = img[y:y+h, x:x+w]

    # Redimensionne le chiffre à 20x20 (comme MNIST)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Crée une image vide 28x28 et place le chiffre au centre
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    return canvas


# ================================
# 📥 Chargement des données MNIST
# ================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape → (nombre, hauteur, largeur, canaux)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Vérification des dimensions
print(x_test.shape)
print(x_train.shape)

# Normalisation des pixels entre [0,1]
x_train = x_train / 255
x_test = x_test / 255


# ================================
# 🖼️ Ajout d'images locales perso
# ================================
# Chemins + labels des images locales
paths = [
    ("images/mon_chiffre_1.png", 1), ("images/mon_chiffre_2.png", 2), ("images/mon_chiffre_3.png", 3),
    ("images/mon_chiffre_4.png", 4), ("images/mon_chiffre_5.png", 5), ("images/mon_chiffre_6.jpg", 6),
    ("images/mon_chiffre_7.png", 7), ("images/mon_chiffre_8.png", 8), ("images/mon_chiffre_9.png", 9)
]

X_custom = []  # Images custom préparées
y_custom = []  # Labels associés

for path, label in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Chargement en niveaux de gris
    if img is None:
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    img = center_and_resize(img)  # Centrage et redimensionnement
    img = 255 - img  # Inversion (fond noir / chiffre blanc)
    img = img.astype('float32') / 255.0  # Normalisation
    img = img.reshape(28, 28, 1)  # Reshape pour CNN
    X_custom.append(img)
    y_custom.append(label)

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)


# ================================
# 🔄 Augmentation de données sur images locales
# ================================
datagen = ImageDataGenerator(
    rotation_range=10,  # Petites rotations
    zoom_range=0.1,  # Zoom léger
    width_shift_range=0.1,  # Décalage horizontal
    height_shift_range=0.1  # Décalage vertical
)

X_aug, y_aug = [], []

# Pour chaque chiffre local → créer 30 variations
for img, label in zip(X_custom, y_custom):
    img = img.reshape(1, 28, 28, 1)
    count = 0
    for batch in datagen.flow(img, batch_size=1):
        X_aug.append(batch[0])
        y_aug.append(label)
        count += 1
        if count >= 30:  # Stop après 30 augmentations
            break

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)


# ================================
# 🔀 Fusion MNIST + images locales
# ================================
# Ajout des images custom + augmentées dans le dataset d'entraînement
x_train_combined = np.concatenate([x_train, X_custom, X_aug], axis=0)
y_train_combined = np.concatenate([y_train, y_custom, y_aug], axis=0)

# Mélange des données pour éviter un biais
from sklearn.utils import shuffle
x_train_combined, y_train_combined = shuffle(x_train_combined, y_train_combined, random_state=42)


# ================================
# 🧠 Création du modèle CNN
# ================================
model = Sequential()

# Bloc convolution + pooling
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))  # Dropout pour éviter le surapprentissage

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))

# Flatten → passage en vecteur
model.add(Flatten())

# Dense layer fully connected
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Couche de sortie (10 classes)
model.add(Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# ================================
# 💾 Callbacks : sauvegarde du meilleur modèle
# ================================
checkpoint = ModelCheckpoint(
    'Save/temp_mnist_cnn_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ================================
# 🚀 Entraînement du modèle
# ================================
history = model.fit(
    x_train_combined, y_train_combined,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[checkpoint]
)


# ================================
# 📊 Courbes d'entraînement
# ================================
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title("Évolution de la précision")
plt.show()


# ================================
# 📈 Évaluation & comparaison modèle
# ================================
from sklearn.metrics import roc_auc_score
import os

# Fonction AUC
def calcul_auc(model, x_test, y_test):
    y_proba = model.predict(x_test)
    return roc_auc_score(y_test, y_proba, multi_class='ovr')

# Comparer modèle nouveau vs ancien
def charger_et_comparer_models(nouveau_path, ancien_path, x_test, y_test):
    if not os.path.exists(nouveau_path):
        raise FileNotFoundError("❌ Aucun modèle temporaire trouvé après l'entraînement.")

    print("\n📥 Chargement du modèle nouvellement entraîné...")
    nouveau_modele = load_model(nouveau_path)
    auc_nouveau = calcul_auc(nouveau_modele, x_test, y_test)

    if os.path.exists(ancien_path):
        print("📥 Chargement de l'ancien modèle...")
        ancien_modele = load_model(ancien_path)
        auc_ancien = calcul_auc(ancien_modele, x_test, y_test)

        print(f"\n🎯 Ancien AUC : {auc_ancien:.3f}")
        print(f"🚀 Nouveau AUC : {auc_nouveau:.3f}")

        if auc_nouveau > auc_ancien:
            print("✅ Nouveau modèle meilleur → Remplacement effectué.")
            nouveau_modele.save(ancien_path)
            return nouveau_modele
        else:
            print("❌ Ancien modèle conservé (meilleur AUC).")
            return ancien_modele
    else:
        print("📁 Aucun modèle précédent. Le modèle actuel devient le meilleur.")
        nouveau_modele.save(ancien_path)
        return nouveau_modele

# Charger le meilleur modèle
meilleur_modele = charger_et_comparer_models(
    "Save/temp_mnist_cnn_model.keras",
    "Save/mnist_cnn_model.keras",
    x_test,
    y_test
)

# Évaluation finale
meilleur_modele.evaluate(x_test, y_test)


# ================================
# 🔍 Fonction : préparation des images locales pour prédiction
# ================================
def load_and_prepare_image(filepath):
    """
    Charge une image, la centre, la redimensionne, l'inverse, la normalise
    et la reshape pour la prédiction.
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Chargement en gris
    if img is None:
        raise ValueError(f"Image non trouvée ou illisible : {filepath}")
    img = center_and_resize(img)  # Centrage/redimensionnement
    img = 255 - img  # Inversion (fond noir)
    img = img.astype('float32') / 255.0  # Normalisation
    img = img.reshape(1, 28, 28, 1)  # Reshape pour modèle
    return img


# Charger les images locales pour tester
img1 = load_and_prepare_image("images/mon_chiffre_1.png")
img2 = load_and_prepare_image("images/mon_chiffre_2.png")
img3 = load_and_prepare_image("images/mon_chiffre_3.png")
img4 = load_and_prepare_image("images/mon_chiffre_4.png")
img5 = load_and_prepare_image("images/mon_chiffre_5.png")
img6 = load_and_prepare_image("images/mon_chiffre_6.png")
img7 = load_and_prepare_image("images/mon_chiffre_7.png")
img8 = load_and_prepare_image("images/mon_chiffre_8.png")
img9 = load_and_prepare_image("images/mon_chiffre_9.png")


# ================================
# 🎨 Affichage + prédictions locales
# ================================
for i, (img, label) in enumerate(zip([img1, img2, img3, img4, img5, img6, img7, img8, img9], range(1, 10))):
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Image perso {label}")
    plt.axis('off')
    plt.show()

    pred = meilleur_modele.predict(img)
    print(f"Class prédite pour {label} :", np.argmax(pred))


# ================================
# 🔄 Fine-tuning avec les images locales
# ================================
# On combine les 9 chiffres locaux
X_custom = np.concatenate([img1, img2, img3, img4, img5, img6, img7, img8, img9], axis=0)
y_custom = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# On entraîne quelques epochs supplémentaires (pour adapter au style perso)
meilleur_modele.fit(X_custom, y_custom, epochs=5, batch_size=2, verbose=1)

# Sauvegarde du modèle amélioré
meilleur_modele.save('Save/mnist_cnn_model.keras')
