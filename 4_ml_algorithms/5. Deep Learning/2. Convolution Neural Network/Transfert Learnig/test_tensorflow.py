import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical


# ------------------
# 1. Data
# ------------------
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 100)
y_test  = to_categorical(y_test, 100)

# Resize plus petit pour CPU (96x96)
x_train = tf.image.resize(x_train, (128, 128))
x_test  = tf.image.resize(x_test, (128, 128))

# ------------------
# 2. Modèle
# ------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False  # geler tout au début

# Construire le modèle
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(100, activation='softmax')   # CIFAR-100
])


# ------------------
# 2. Étape 1 : Entraînement de la tête seule
# ------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # LR normal
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', "auc"])

print("🔹 Phase 1: Entraînement de la tête seule")
history_head = model.fit(x_train, y_train,
                         batch_size=32,
                         epochs=10,
                         validation_data=(x_test, y_test))

# ------------------
# 3. Étape 2 : Fine-tuning (dé-gel partiel)
# ------------------
# Dégeler les 20 dernières couches du backbone
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompiler avec un LR beaucoup plus petit
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'auc'])

from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


print("🔹 Phase 2: Fine-tuning des dernières couches")
history_finetune = model.fit(x_train, y_train,
                             batch_size=64,
                             epochs=20,
                             validation_data=(x_test, y_test),
                             callbacks=[lr_reduce, early_stop])

loss, accuracy, auc = model.evaluate(x_test, y_test, verbose=1)
print(f" AUC : {auc:.4f}")
print(f"Precision du model de base : {accuracy:.4f}")
print(f"la perte du model de base : {loss:.4f}")




# Sauvegarder le modèle complet
model.save("Save/mobilenetv2_cifar100.keras")

# Pour recharger
from tensorflow.keras.models import load_model
model = load_model("Save/mobilenetv2_cifar100.keras")

# Sauvegarder uniquement les poids
model.save_weights("Save/mobilenetv2_cifar100.weights.h5")

# Pour recharger
model.load_weights("Save/mobilenetv2_cifar100.weights.h5")

