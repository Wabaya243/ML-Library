import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
import numpy as np

import matplotlib.pyplot as plt

# Chargement du meilleur modèle
from sklearn.metrics import roc_auc_score
import os

def calcul_auc(model, x_test, y_test):
    y_proba = model.predict(x_test)
    return roc_auc_score(y_test, y_proba, multi_class='ovr')

def charger_et_comparer_models(nouveau_path, ancien_path, x_test, y_test, tol=1e-5):
    if not os.path.exists(nouveau_path):
        raise FileNotFoundError(" Aucun modèle temporaire trouvé après l'entraînement.")

    print("\n Chargement du modèle nouvellement entraîné...")
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

        # Comparaison AUC
        if auc_nouveau > auc_ancien + tol:
            print(" Nouveau modèle meilleur en AUC → Remplacement effectué.")
            nouveau_modele.save(ancien_path)
            return nouveau_modele
        # Si AUC identique, comparer précision
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



#On charge les données des cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#on normalise les valeurs pixels a 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# One-hot encoding pour les valeurs des sortie pour que les valeurs soit exacte a l'entre de la convolution
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



print(f"Training data shape: {x_train.shape}, {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, {y_test.shape}")

#On cree les models

model = Sequential([
    
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)), # 1er Couche des Convolution
    BatchNormalization(),
    MaxPooling2D(2,2), # Un couhe de Pooling pour affiner
    Dropout(0.1),
    
    Conv2D(64,(3,3), activation = 'relu', padding='same'), # 2eme couche des convolution pour detecter encore plus d'info
    BatchNormalization(),
    MaxPooling2D(2,2), # un couche de pooling
    Dropout(0.1),
   
    Conv2D(128,(3,3), activation = 'relu', padding='same'), # 2eme couche des convolution pour detecter encore plus d'info
    BatchNormalization(),
    MaxPooling2D(2,2), # un couche de pooling
    Dropout(0.3),
    
    
    Conv2D(256,(3,3), activation = 'relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),
    
    Flatten(), # le couhe de flattening pour transformer les vecteurs 2D en vecteurs 1D
    Dense(256, activation='relu'), # La premier couche du reseau de neurones fully connected
    Dropout(0.4), # pour eviter les surapprentisage
    Dense(10, activation='softmax') # la couche de sortie avec softmax pour classification
    
    ])



lr_schedule_simple = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.9,
)

optimizerW_simple = AdamW(learning_rate=lr_schedule_simple, weight_decay=1e-4)



# On compile les models
model.compile(optimizer=optimizerW_simple, loss = 'categorical_crossentropy', metrics=['accuracy'])


# Afficher les info (caracteristiques) du model
model.summary()


# pour sauvegarder
checkpoint_simple = ModelCheckpoint('Save/temp_cifar10_cnn_model_simple.keras', monitor='val_accuracy', save_best_only=True, verbose=1)


early_stopping_test = EarlyStopping(
    monitor='val_loss',
    patience=4,  # Arrête si val_loss ne s'améliore pas après 4 epochs
    restore_best_weights=True
)

reduce_lr_simple = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)


#entraine les models

history = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = 30,
    batch_size= 32,
    verbose = 1,
    callbacks=[checkpoint_simple, early_stopping_test]
    )


#On charge les meilleurs models
meilleur_modele_simple = charger_et_comparer_models(
    "Save/temp_cifar10_cnn_model_simple.keras",
    "Save/cifar10_meilleur_model_simple.keras",
    x_test,
    y_test
)

# Evaluer les models

loss, accuracy = meilleur_modele_simple.evaluate(x_test, y_test, verbose=0)
print(f"Precision du model de base : {accuracy:.4f}")
print(f"la perte du model de base : {loss:.4f}")






import shutil

def prepare_log_dir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

log_dir = r"C:\tensorboard_logs\cifar10\improved"
prepare_log_dir(log_dir)

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)




#La Version du model amelioerer 

from sklearn.model_selection import train_test_split

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    shear_range=0.0,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)


# Split manuel
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# Générateur uniquement sur les données d'entraînement
train_generator = datagen.flow(x_train_split, y_train_split, batch_size=64, seed=42)

val_generator = ImageDataGenerator().flow(x_val_split, y_val_split, batch_size=64)



datagen_test = ImageDataGenerator()
train_generator_test = datagen_test.flow(x_train_split, y_train_split, batch_size=64)
images_test, labels_test = next(iter(train_generator_test))

plt.imshow(images_test[0])
plt.title(f"Label: {np.argmax(labels_test[0])}")
plt.show()



early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,  # Arrête si val_loss ne s'améliore pas après 3 epochs
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6) 

# Nombre total de batchs par epoch
steps_per_epoch = len(x_train_split) // 64  # batch_size = 32

# Cosine decay sur tout l'entraînement
lr_schedule_cosin = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01,
    decay_steps=steps_per_epoch * 50  # 80 epochs
)

# SGD avec le schedule
optimizer_sgd = tf.keras.optimizers.SGD(
    learning_rate=lr_schedule_cosin,
    momentum=0.9,
    nesterov=True
)


optimizerW = AdamW(learning_rate=0.001, weight_decay=1e-4)
optimizer = Adam(learning_rate=0.001)  # Essaie 0.0005, 0.0001 aussi




#creation du modele
improved_model = Sequential()

# L'ajout du convolution layer
improved_model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
improved_model.add(BatchNormalization())
improved_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(MaxPooling2D((2,2)))
improved_model.add(Dropout(0.05))

improved_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(MaxPooling2D((2,2)))
improved_model.add(Dropout(0.05))

improved_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(MaxPooling2D((2,2)))
improved_model.add(Dropout(0.1))

improved_model.add(Conv2D(256,(3,3), activation = 'relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
improved_model.add(BatchNormalization())
improved_model.add(Dropout(0.15))


# Ajout de la phase de flattening
improved_model.add(Flatten())


improved_model.add(Dense(256, activation='relu'))
improved_model.add(Dropout(0.2))

# Ajout de la phase de full connection
improved_model.add(Dense(128, activation='relu'))
improved_model.add(Dropout(0.15))
# Ajout de la phase de la sortie
improved_model.add(Dense(10, activation='softmax'))

improved_model.summary()

# Compilation du modele
improved_model.compile(optimizer=optimizer_sgd, loss='categorical_crossentropy' , metrics=['accuracy'])

#pour sauvegarder automatiquement pendant l'entrainement
checkpoint = ModelCheckpoint('Save/temp_cifar10_cnn_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Entrainement du modele
improved_history = improved_model.fit( 
    train_generator,
    epochs=80,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint, tensorboard],
    verbose=1,
  )



#On charge les meilleurs models
meilleur_modele = charger_et_comparer_models(
    "Save/temp_cifar10_cnn_model.keras",
    "Save/cifar10_meilleur_model.keras",
    x_test,
    y_test
)



# evaluation du modele
# Evaluer les nouveau modele ameliorer

improved_loss,improved_accuracy = meilleur_modele.evaluate(x_test, y_test, verbose=0)
print(f"Precision du model ameliorer : {improved_accuracy:.4f}")
print(f"la perte du model ameliorer : {improved_loss:.4f}")





from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Dictionnaire pour décoder les prédictions (CIFAR-10 labels)
cifar10_labels = {
    0: 'Avion',
    1: 'Voiture',
    2: 'Oiseau',
    3: 'Chat',
    4: 'Cerf',
    5: 'Chien',
    6: 'Grenouille',
    7: 'Cheval', 
    8: 'Bateau',
    9: 'Camion'
}

def charger_et_predire_image(path_image, modele):
    img = image.load_img(path_image, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    prediction = modele.predict(img_array)
    classe_predite = np.argmax(prediction)

    # Affichage
    plt.imshow(img)
    plt.title(f"Classe prédite : {cifar10_labels[classe_predite]}")
    plt.axis('off')
    plt.show()

    return classe_predite


# chemin vers ton image locale
chemin_image = "chemin/vers/ton_image.jpg"  # <-- modifie ce chemin
charger_et_predire_image(chemin_image, meilleur_modele)
charger_et_predire_image(chemin_image, meilleur_modele_simple)



from sklearn.metrics import f1_score

# après évaluation
y_pred = meilleur_modele.predict(x_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
y_true_classes = tf.argmax(y_test, axis=1)

f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
print(f"F1 Score : {f1:.4f}")


# Plot training and validation accuracy
plt.plot(improved_history.history['accuracy'], label="Training Accuracy")
plt.plot(improved_history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(improved_history.history['loss'], label="Training Loss")
plt.plot(improved_history.history['val_loss'], label="Validation Loss")
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.imshow(x_train[1])
plt.title(f'CIfar10  nom: {y_train[1]} ')
plt.show()
