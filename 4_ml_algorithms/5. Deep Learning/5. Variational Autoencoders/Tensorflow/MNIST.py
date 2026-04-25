# ===============================
# Importations
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.datasets import mnist, fashion_mnist

# ===============================
# Paramètres globaux
# ===============================
num_features = 784
batch_size = 128
epochs = 30
hidden_1 = 129
hidden_2 = 64

# ===============================
# Fonction de chargement des données
# ===============================
def load_data(choice="mnist", labels=False):
    if choice not in ["mnist", "fashion_mnist"]:
        raise ValueError("veuillez choisir entre mnist et fashion_mnist")

    if choice == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    X_train, X_test = x_train / 255.0, x_test / 255.0
    X_train, X_test = X_train.reshape([-1, num_features]), X_test.reshape([-1, num_features])
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    if labels:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, X_test

# ===============================
# Fonctions utilitaires
# ===============================
def plot_prediction(y_true, y_pred):
    f, ax = plt.subplots(2, 10, figsize=(15, 4))
    n = min(10, y_true.shape[0])
    for i in range(n):
        ax[0][i].imshow(np.reshape(y_true[i], (28, 28)), cmap="gray")
        ax[0][i].axis("off")
        ax[1][i].imshow(np.reshape(y_pred[i], (28, 28)), cmap="gray")
        ax[1][i].axis("off")
    plt.tight_layout()
    plt.show()

def plot_digit(X, y, encoder, batch_size=128):
    preds = encoder.predict(X, batch_size=batch_size)
    z_mean = preds[0] if isinstance(preds, (list, tuple)) else preds
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y)
    plt.colorbar()
    plt.xlabel("z[0] latent Dimension")
    plt.ylabel("z[1] latent Dimension")
    plt.show()

def generate_manifold(decoder, n=15):
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size:(i + 1) * digit_size,
                   j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.xlabel("z[0] Latent Dimension")
    plt.ylabel("z[1] Latent Dimension")
    plt.show()

# ===============================
# AE simple
# ===============================
ae_inputs = kr.Input(shape=(num_features,))
x = kr.layers.Dense(hidden_1, activation="sigmoid")(ae_inputs)
encoded = kr.layers.Dense(hidden_2, activation="sigmoid")(x)
ae_encoder_model = kr.Model(ae_inputs, encoded, name="ae_encoder")

latent_inputs = kr.Input(shape=(hidden_2,))
x = kr.layers.Dense(hidden_1, activation="sigmoid")(latent_inputs)
decoded = kr.layers.Dense(num_features, activation="sigmoid")(x)
ae_decoder_model = kr.Model(latent_inputs, decoded, name="ae_decoder")

ae_outputs = ae_decoder_model(ae_encoder_model(ae_inputs))
mnist_ae_model = kr.Model(ae_inputs, ae_outputs, name='mnist_ae')
mnist_ae_model.compile(optimizer="adam", loss="mse")

X_train, X_test = load_data("mnist")
mnist_ae_model.fit(X_train, X_train,
                   epochs=epochs, batch_size=batch_size,
                   validation_data=(X_test, X_test))

y_true = X_test[:10]
y_pred = mnist_ae_model.predict(y_true)
plot_prediction(y_true, y_pred)

# ===============================
# VAE Dense en subclassing
# ===============================
def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_log_var))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

hidden_dim = 512
latent_dim = 2

vae_inputs = kr.Input(shape=(num_features,))
x = kr.layers.Dense(hidden_dim, activation='relu')(vae_inputs)
z_mean = kr.layers.Dense(latent_dim)(x)
z_log_var = kr.layers.Dense(latent_dim)(x)
z = kr.layers.Lambda(sampling)([z_mean, z_log_var])
vae_encoder = kr.Model(vae_inputs, [z_mean, z_log_var, z])

latent_inputs = kr.Input(shape=(latent_dim,))
x = kr.layers.Dense(hidden_dim, activation='relu')(latent_inputs)
vae_outputs = kr.layers.Dense(num_features, activation='sigmoid')(x)
vae_decoder = kr.Model(latent_inputs, vae_outputs)

class VAE(kr.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}

X_train, y_train, X_test, y_test = load_data("mnist", labels=True)
vae = VAE(vae_encoder, vae_decoder)
vae.compile(optimizer="adam")
vae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, X_test))

generate_manifold(vae_decoder)
plot_digit(X_test, y_test, vae_encoder)

# ===============================
# ConvVAE en subclassing
# ===============================
input_shape = (28, 28, 1)
filters = 32
latent_dim = 2

conv_inputs = kr.Input(shape=input_shape)
x = kr.layers.Conv2D(filters, 3, activation='relu', strides=2, padding='same')(conv_inputs)
x = kr.layers.Conv2D(filters*2, 3, activation='relu', strides=2, padding='same')(x)
shape = tf.keras.backend.int_shape(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(16, activation='relu')(x)
z_mean = kr.layers.Dense(latent_dim)(x)
z_log_var = kr.layers.Dense(latent_dim)(x)
z = kr.layers.Lambda(sampling)([z_mean, z_log_var])
conv_encoder = kr.Model(conv_inputs, [z_mean, z_log_var, z])

latent_inputs = kr.Input(shape=(latent_dim,))
x = kr.layers.Dense(shape[1]*shape[2]*shape[3], activation='relu')(latent_inputs)
x = kr.layers.Reshape((shape[1], shape[2], shape[3]))(x)
x = kr.layers.Conv2DTranspose(filters*2, 3, strides=2, padding='same', activation='relu')(x)
x = kr.layers.Conv2DTranspose(filters, 3, strides=2, padding='same', activation='relu')(x)
conv_outputs = kr.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
conv_decoder = kr.Model(latent_inputs, conv_outputs)

class ConvVAE(kr.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2, 3]))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}

X_train_conv = X_train.reshape([-1, 28, 28, 1])
X_test_conv = X_test.reshape([-1, 28, 28, 1])

conv_vae = ConvVAE(conv_encoder, conv_decoder)
conv_vae.compile(optimizer="adam")
conv_vae.fit(X_train_conv, X_train_conv, epochs=epochs,
             batch_size=batch_size, validation_data=(X_test_conv, X_test_conv))

generate_manifold(conv_decoder)
plot_digit(X_test, y_test, conv_encoder)
