import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from autoencoder import VAE
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalization and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, y_train, x_test, y_test

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, frames, 1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # (3000, 256, 64, 1)
    return x_train

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(64, 256, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernel=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder

if __name__ == '__main__':


    x_train = load_fsdd("spectrograms/")
    print(x_train.shape)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    # Load the model
    autoencoder2 = VAE.load("model")
    autoencoder2.summary()

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train[0:500, :, :, :]))
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
    # print(train_dataset)

    # val_dataset = tf.data.Dataset.from_tensor_slices((x_test[0:500:, :, :, :]))
    # val_dataset = val_dataset.batch(BATCH_SIZE)

    # # metrics
    # train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    # val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # autoencoder = VAE(
    #     input_shape=(28, 28, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernel=(3, 3, 3, 3),
    #     conv_strides=(1, 2, 2, 1),
    #     latent_space_dim=2
    # )
    # autoencoder.compile(LEARNING_RATE)


    # print("EMPIEZA EL ENTRENAMIENTO")

    # loss =  MeanSquaredError()
    # # optimizer = Adam(learning_rate=LEARNING_RATE)
    # optimizer = SGD(lr=LEARNING_RATE)

    # model = autoencoder.model
    # model.summary()


    # for epoch in range(EPOCHS):
    #     print(f'Epoch: {epoch}')
    #     for step, x_batch_train in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             logits = model(x_batch_train, training=True)
    #             loss_value = loss(x_batch_train, logits)

    #         # print("AQUIIII")
    #         grads = tape.gradient(loss_value, model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #         train_acc_metric.update_state(x_batch_train, logits)
    #         # print(train_acc_metric.result())
    #         # print(
    #         #     "Training loss (for one batch) at step %d: %.4f"
    #         #     % (step, float(loss_value))
    #         # )
    #         # print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))

    #     # Display metrics at the end of each epoch.
    #     train_acc = train_acc_metric.result()
    #     print("Training acc over epoch: %.4f" % (float(train_acc),))
    #     # Reset training metrics at the end of each epoch
    #     train_acc_metric.reset_states()

    #     # Run a validation loop at the end of each epoch.
    #     for x_batch_val in val_dataset:
    #         val_logits = model(x_batch_val, training=False)
    #         # Update val metrics
    #         val_acc_metric.update_state(x_batch_val, val_logits)
    #     val_acc = val_acc_metric.result()
    #     val_acc_metric.reset_states()
    #     print("Validation acc: %.4f" % (float(val_acc),))
