#pour integrer dans une App Mobile 

import tensorflow as tf

model = tf.keras.models.load_model("Save/cifar10_meilleur_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("Exportation/model.tflite", "wb") as f:  # => dans le dossier Save/
    f.write(tflite_model)
    
    
    
    
# en app web

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image

# Dictionnaire de labels CIFAR-10
cifar10_labels = {
    0: 'Avion', 1: 'Voiture', 2: 'Oiseau', 3: 'Chat', 4: 'Cerf',
    5: 'Chien', 6: 'Grenouille', 7: 'Cheval', 8: 'Bateau', 9: 'Camion'
}

# Charger le modèle
model = load_model("Save/cifar10_meilleur_model.keras")

st.title("🎯 Prédicteur d'image - CIFAR-10")

# Upload image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).resize((32, 32))
    st.image(img, caption="Image chargée", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)

    st.write(f"🧠 Prédiction : **{cifar10_labels[pred_index]}**")

