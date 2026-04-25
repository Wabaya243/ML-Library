from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import os

# -------------------------
#  Fonction AUC
# -------------------------
def calcul_auc(model, x_test, y_test):
    y_proba = model.predict(x_test, verbose=0)
    return roc_auc_score(y_test, y_proba, multi_class='ovr')

# -------------------------
#  Comparaison & sauvegarde du meilleur modèle
# -------------------------
def charger_et_comparer_models(nouveau_path, ancien_path, x_test, y_test, tol=1e-3):
    if not os.path.exists(nouveau_path):
        raise FileNotFoundError(" Aucun modèle temporaire trouvé après l'entraînement.")

    print("\n📥 Chargement du modèle nouvellement entraîné...")
    nouveau_modele = load_model(nouveau_path)
    auc_nouveau = calcul_auc(nouveau_modele, x_test, y_test)
    loss_nouveau, acc_nouveau = nouveau_modele.evaluate(x_test, y_test, verbose=0)

    if os.path.exists(ancien_path):
        print(" Chargement de l'ancien modèle...")
        ancien_modele = load_model(ancien_path)
        auc_ancien = calcul_auc(ancien_modele, x_test, y_test)
        loss_ancien, acc_ancien = ancien_modele.evaluate(x_test, y_test, verbose=0)

        print(f"\n Ancien AUC : {auc_ancien:.5f} | Ancienne précision : {acc_ancien:.5f}")
        print(f" Nouveau AUC : {auc_nouveau:.5f} | Nouvelle précision : {acc_nouveau:.5f}")

        if auc_nouveau > auc_ancien + tol:
            print(" Nouveau modèle meilleur en AUC → Remplacement effectué.")
            nouveau_modele.save(ancien_path)
            return nouveau_modele
        elif abs(auc_nouveau - auc_ancien) <= tol and acc_nouveau > acc_ancien + tol:
            print(" AUC identique mais précision meilleure → Remplacement effectué.")
            nouveau_modele.save(ancien_path)
            return nouveau_modele
        else:
            print(" Ancien modèle conservé (aucune amélioration significative).")
            return ancien_modele
    else:
        print(" Aucun modèle précédent. Le modèle actuel devient le meilleur.")
        nouveau_modele.save(ancien_path)
        return nouveau_modele


# -------------------------
#  Prétraitement des données
# -------------------------
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

nb_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

# Visualisation
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
plt.show()

# -------------------------
#  Définition du modèle CNN amélioré
# -------------------------
model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))

model.summary()

# -------------------------
#  Compilation
# -------------------------
model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(learning_rate=1e-4, decay=1e-6),
    metrics=['accuracy']
)

# -------------------------
#  Data Augmentation
# -------------------------
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1
)
datagen.fit(X_train)

# -------------------------
#  Callbacks
# -------------------------
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True, verbose=1)

checkpoint_simple = ModelCheckpoint("Save/temp_cifar100_cnn_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)

reduce_lr_simple = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# -------------------------
#  Entraînement
# -------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=50),
    steps_per_epoch=X_train.shape[0] // 50,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint_simple, reduce_lr_simple],
    verbose=1
)

# -------------------------
#  Chargement du meilleur modèle
# -------------------------
meilleur_modele = charger_et_comparer_models(
    "Save/temp_cifar100_cnn_model.keras",
    "Save/cifar100_meilleur_model.keras",
    X_test, y_test
)

# -------------------------
#  Évaluation finale
# -------------------------
improved_loss, improved_accuracy = meilleur_modele.evaluate(X_test, y_test, verbose=1)
print(f" Précision finale : {improved_accuracy:.4f}")
print(f" Perte finale : {improved_loss:.4f}")

# -------------------------
#  Visualisation des courbes
# -------------------------
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend(); plt.title("Évolution de la perte"); plt.show()

plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.legend(); plt.title("Évolution de la précision"); plt.show()


