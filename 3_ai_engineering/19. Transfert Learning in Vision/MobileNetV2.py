############### MobileNetV2 ###########################

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


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





#charger les MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

#gelons les couches
for layer in base_model.layers:
    layer.trainable = False

#ajouter des couches de classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# Augmentons les données
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20, #rotation aléatoire
    width_shift_range=0.2, #décalage horizontal aléatoire
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    'intel_data/seg_train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    'intel_data/seg_test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


#  Conversion en tableaux pour pouvoir calculer AUC correctement
x_val, y_val = [], []
for batch_x, batch_y in val_data:
    x_val.append(batch_x)
    y_val.append(batch_y)
    if len(x_val) >= len(val_data):
        break

x_val = np.vstack(x_val)
y_val = np.vstack(y_val)


# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("Save/temp_MobileNetV2_model.keras", save_best_only=True, monitor='val_accuracy', verbose=1)

# Entraîner le modèle
history = model.fit(
    train_data,
    epochs=10, 
    validation_data=val_data,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data),
    callbacks = [checkpoint]
)


#on charge les meilleur model
meilleur_modele = charger_et_comparer_models(
    "Save/temp_MobileNetV2_model.keras",
    "Save/best_MobileNetV2_model.keras",
    x_val,
    y_val
)



# Évaluer le modèle
loss, accuracy = model.evaluate(val_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")



#predire sur les données non etiquété 
pred_datagen = ImageDataGenerator(rescale=1./255)

pred_data = pred_datagen.flow_from_directory(
    'intel_data/seg_pred',
    target_size=(224, 224),
    batch_size=1,
    class_mode=None,  # Pas de labels
    shuffle=False
)

predictions = model.predict(pred_data)
predicted_classes = predictions.argmax(axis=-1)
print(predicted_classes)

#afficher les classes prédites
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

# Nombre d'images à afficher
n = 20

# Récupérer les chemins des images depuis flow_from_directory
image_paths = pred_data.filepaths

plt.figure(figsize=(15, 10))

for i in range(n):
    img_path = image_paths[i]
    img = image.load_img(img_path, target_size=(224, 224))  # redimensionner comme le modèle
    img_array = image.img_to_array(img) / 255.0

    plt.subplot(4, 5, i+1)  # 4 lignes x 5 colonnes
    plt.imshow(img_array)
    plt.title(class_names[predicted_classes[i]])
    plt.axis('off')

plt.tight_layout()
plt.show()


# Chemin de l'image à prédire
img_path = 'intel_data/seg_pred/mon_image.jpg' 

# Charger et préparer l'image
img = image.load_img(img_path, target_size=(224, 224))  # redimensionne comme le modèle
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # ajouter dimension batch

# Prédire
prediction = model.predict(img_array)
predicted_class = prediction.argmax(axis=-1)[0]

# Afficher l'image et la classe prédite
plt.imshow(img_array[0])
plt.title(class_names[predicted_class])
plt.axis('off')
plt.show()

print(f"Classe prédite : {class_names[predicted_class]}")

