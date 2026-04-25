import cv2
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping 


train_folder= '/kaggle/input/animefacedataset'

files = ['images']

def Display_IMAGES(folder):
    for file in files:
        path = os.path.join(folder, file)
        fig, axes = plt.subplots(1, 3, figsize=(25, 7)) 
        for i, img in enumerate(os.listdir(path)[:3]): 
            img_array = cv2.imread(os.path.join(path, img))
            
            # Convert from BGR to RGB for proper display in matplotlib
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            img_shape = img_rgb.shape
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"IMG_Size: {img_shape}")
            axes[i].axis('off')
            
        plt.tight_layout()
    
    plt.show()

Display_IMAGES(train_folder)

## images do not have the same size
## I will resize them to (64×64) with 3 channeels.


IMAGE_SIZE =  (64,64)



#📌 DATA
print('Training Images:')
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=254
)

### visualiser apres resizing 
def plot_images(dataset, num_images=10):
    plt.figure(figsize=(50, 20))
    for images, _ in dataset.take(1):
        for i in range(num_images):
            plt.subplot(2, 5, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis('off')
    plt.show()

plot_images(train_ds)

X_train = []

for images, _ in train_ds:
    X_train.append(images.numpy())


X_train = np.concatenate(X_train, axis=0)

print(f"X_train shape: {X_train.shape}")

## normalisation
X_train=X_train/255.0

plt.figure(figsize=(12, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)  
    plt.imshow(X_train[i])   
    plt.axis('off')         
  

plt.tight_layout()
plt.show()

latent_dim = 128  

np.random.seed(42)
tf.random.set_seed(42)


#encoder 
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    
    latent_space = layers.Dense(latent_dim, name='latent_space')(x)
    
    return models.Model(inputs, latent_space, name='encoder')

encoder = build_encoder((64, 64, 3), latent_dim)
encoder.summary()

## Decoder

def build_decoder(latent_dim, original_shape):
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    base_dim = original_shape[0] // 8  
    x = layers.Dense(base_dim * base_dim * 128, activation='relu')(latent_inputs)
    x = layers.Reshape((base_dim, base_dim, 128))(x) 
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return models.Model(latent_inputs, outputs, name='decoder')


decoder = build_decoder(latent_dim, (64, 64, 3))
decoder.summary()


### VAE

encoder = build_encoder((64, 64, 3), latent_dim)
decoder = build_decoder(latent_dim, (64, 64, 3))

inputs = layers.Input(shape=(64, 64, 3))
latent = encoder(inputs)
outputs = decoder(latent)

vae = models.Model(inputs, outputs, name='Variational_Autoencoder')

vae.summary()

class ImageGenerator(callbacks.Callback):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        num_images = 10
        random_latent_vectors = np.random.normal(size=(num_images, self.latent_dim))
        generated_images = self.model.layers[2](random_latent_vectors) 

        # images range [0, 1]
        generated_images = tf.clip_by_value(generated_images, 0, 1).numpy()

        plt.figure(figsize=(20, 3))
        for i in range(num_images):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(generated_images[i])
            plt.axis('off')
        plt.suptitle(f'Epoch {epoch + 1}')
        plt.show()


    vae.compile(optimizer='adam', loss='mse')

early_stopping = callbacks.EarlyStopping(
    monitor='loss',
    patience=5, 
    restore_best_weights=True)

vae.fit(
    X_train,
    X_train,
    batch_size=64,
    epochs=100,
    callbacks=[early_stopping, ImageGenerator(latent_dim)]
)



def visualize_original_vs_generated(original_images, generated_images, n=10):
    plt.figure(figsize=(20, 5))
    
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i])
        plt.title("Original")
        plt.axis("off")
        
        # Generated images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(generated_images[i])
        plt.title("\nGenerated\n(VAE)")
        plt.axis("off")
    
    plt.show()


# visualiser les images original et les images generer via VAE
generated_images = vae.predict(X_train)

visualize_original_vs_generated(X_train, generated_images)
