import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, ReLU,
    BatchNormalization, Flatten, Dense,
    Reshape, Conv2DTranspose, Activation,
    Lambda
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

tf.compat.v1.disable_eager_execution()

class VAE():
    '''
    Deep convolutional autoencoder architecture.
    '''
    def __init__(
        self, input_shape, conv_filters, 
        conv_kernel, conv_strides,
        latent_space_dim):
    
        self.input_shape = input_shape# (width, heigth, channels)
        self.conv_filters = conv_filters# e.g (2, 4, 8)
        self.conv_kernels = conv_kernel# e.g. (3, 5, 3)
        self.conv_strides = conv_strides# e.g. (1, 2, 1)
        self.latent_space_dim = latent_space_dim# e.g 5
        
        self.encoder = None
        self.decoder = None
        self.model = None
        self._shape_before_bottleneck = None
        self._model_input = None
        self.reconstruction_loss_weight = 100000

        self.optimizer = None
        self.loss = None
        
        # Number of conv layers
        self._num_conv_layers = len(self.conv_filters)
        # Build the model
        self._build()

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_true, y_pred):
        reconstruction_loss = self._calculate_reconstruction_loss(y_true, y_pred)
        kl_loss = self._calculate_kl_loss(y_true, y_pred)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_true, y_pred):
        error = y_true - y_pred
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_true, y_pred):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def compile(self, learning_rate=0.001):
        self.optimizer = Adam(learning_rate=learning_rate)
        # self.loss = MeanSquaredError()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self._calculate_combined_loss
        )
        
    def train(self, x_train, batch_size, n_epochs):
        self.model.fit(
            x_train, 
            x_train,
            batch_size=batch_size,
            epochs=n_epochs,
            shuffle=True
        )
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
        
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")
        
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
        
    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        # print(f"BOTTLENECH SHAPE : {self._shape_before_bottleneck}")
        # print(f"DENSE: {dense_layer}")
        reshaped = Reshape(target_shape=self._shape_before_bottleneck)(dense_layer)
        # print(reshaped)
        return reshaped
        
    def _add_conv_transpose_layers(self, x):
        for layer_idx in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_idx, x)
        return x
        
    def _add_conv_transpose_layer(self, layer_idx, x):
        layer_number = self._num_conv_layers - layer_idx
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_idx],
            kernel_size=self.conv_kernels[layer_idx],
            strides=self.conv_strides[layer_idx],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_number}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_number}")(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=self.input_shape[-1], # (24, 24, 1)
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        x = Activation("sigmoid", name="sigmoid_layer")(x)
        return x
        
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottle_neck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottle_neck, name="encoder")
        
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="inputs")
    
    def _add_conv_layers(self, encoder_input):
        '''Creates all conv blocks'''
        x = encoder_input
        for layer_idx in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_idx, x)
        return x
    
    def _add_conv_layer(self, layer_idx, x):
        '''Convolutional block composed of: Conv2D, RELU and Batch
        normalization'''
        layer_number = layer_idx + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_idx],
            kernel_size=self.conv_kernels[layer_idx],
            strides=self.conv_strides[layer_idx],
            padding="same",
            name=f"encoder_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
        
    def _add_bottleneck(self, x):
        '''Flatten data (Dense layer)'''
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        # x = Dense(self.latent_space_dim, name="encoder_output")(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        def sample_point_normal_dist(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.0, stddev=1.0)

            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point
            
        x = Lambda(sample_point_normal_dist, name="encoder_output")([self.mu, self.log_variance])
        return x
    
if __name__=="__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernel=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
