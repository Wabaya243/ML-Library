# %% [markdown]
# # Reconnaissance Chien vs Chat avec TensorFlow / Keras
# Ce laboratoire montre comment construire un CNN from scratch pour distinguer les chiens des chats.

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# %%
# 1. Paramètres généraux
BATCH_SIZE = 32
IMG_SIZE = (64, 64)  # redimensionner les images à 150x150 (peut varier)
EPOCHS = 20
SEED = 123

# %%
# 2. Chargement des données
# Suppose que les données sont dans un dossier avec sous-dossiers 'cats' et 'dogs' pour l'entraînement
# Par exemple : data/train/cats, data/train/dogs, data/validation/cats, data/validation/dogs

train_dir = 'data/train'
validation_dir = 'data/test'

train_dataset = keras.preprocessing.image_dataset_from_directory(
    train_dir,              # chemin du dossier train/
    labels='inferred',      # les labels (cats/dogs) sont déduits des noms des sous-dossiers
    label_mode='binary',    # sortie = 0 ou 1 (plutôt que "one-hot" ou "categorical")
    batch_size=BATCH_SIZE,  # nombre d'images envoyées au modèle en une fois
    image_size=IMG_SIZE,    # redimensionnement automatique (ex: 64x64 ou 150x150)
    shuffle=True,           # mélanger les images (important pour l'entraînement)
    seed=SEED               # graine aléatoire pour la reproductibilité
)

validation_dataset = keras.preprocessing.image_dataset_from_directory(
    validation_dir,         # chemin du dossier validation/
    labels='inferred',      # idem : labels = noms des sous-dossiers
    label_mode='binary',    # classification binaire
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,          # ici on ne mélange pas (utile pour évaluer)
    seed=SEED
)

# %%
# 3. Prétraitement et data augmentation

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Normalisation des images dans [0,1]
normalization_layer = layers.Rescaling(1./255)

# Appliquer augmentation → normalisation
def prepare(ds, shuffle=False, augment=False):
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y))
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare(train_dataset, shuffle=True, augment=True)
val_ds = prepare(validation_dataset, shuffle=False, augment=False)

# %%
# 4. Construction du modèle CNN

def make_model(input_shape=IMG_SIZE + (3,)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # sortie binaire
    ])
    return model

model = make_model()
model.summary()

# %%
# 5. Compilation

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# %%
# 6. Callbacks

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

# %%
# 7. Entraînement

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

# %%
# 8. Visualisation des courbes d’apprentissage

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training vs Validation Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation Loss')
    plt.show()

plot_history(history)

# %%
# 9. Évaluation plus poussée : matrice de confusion

# Obtenir les prédictions sur le jeu validation
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    # preds sont entre 0 et 1 → convertir en 0 ou 1
    y_pred.extend((preds > 0.5).astype(int).flatten())

print(classification_report(y_true, y_pred, target_names=['cats','dogs']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['cats','dogs'])
plt.yticks(tick_marks, ['cats','dogs'])

# ajouter les nombres
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %%
# 10. Tester sur des images externes

def predict_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # ajouter batch dimension
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    print(f"Prédiction (0=cats, 1=dogs): {prediction[0][0]:.4f}")
    if prediction[0][0] > 0.5:
        print("→ C’est un chien !")
    else:
        print("→ C’est un chat !")

# Exemples
# predict_image('some_path_to_cat.jpg')
# predict_image('some_path_to_dog.jpg')

