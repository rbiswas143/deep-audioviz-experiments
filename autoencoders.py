from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Cropping2D, Reshape
from keras.models import Model, Sequential
from keras import regularizers
import math
from kapre.time_frequency import Spectrogram

# Single layered autoencoder
def basic_ae(encoding_dim, input_size):
    # autoencoder
    inp = Input(shape=(input_size,))
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(inp)
    decoded = Dense(input_size, activation='sigmoid')(encoded)
    autoencoder = Model(inp, decoded)

    # encoder
    encoder = Model(inp, encoded)

    # decoder
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder, encoder, decoder


# Deep autoencoder
def deep_ae(input_size):
    # autoencoder
    inp = Input(shape=(input_size,))

    encoded = Dense(20, activation='relu', activity_regularizer=regularizers.l1(10e-5))(inp)
    encoded = Dense(15, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoded = Dense(5, activation='relu')(encoded)

    decoded = Dense(5, activation='relu')(encoded)
    decoded = Dense(10, activation='relu')(decoded)
    decoded = Dense(15, activation='relu')(decoded)
    decoded = Dense(20, activation='relu')(decoded)
    decoded = Dense(input_size, activation='sigmoid')(decoded)

    autoencoder = Model(inp, decoded)

    # encoder
    encoder = Model(inp, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder


# Convoluted deep autoencoder
def conv_ae_bkup(inpdimx, inpdimy):
    inp = Input(shape=(inpdimx, inpdimy, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inp, decoded)

    # encoder
    encoder = Model(inp, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder


# Convoluted deep autoencoder
def conv_ae(inpdimx, inpdimy):
    inp = Input(shape=(inpdimx, inpdimy, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(1e-5))(x)

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inp, decoded)

    # encoder
    encoder = Model(inp, encoded)

    autoencoder.compile(optimizer='adadelta',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    return autoencoder, encoder

def kapre_conv_ae(num_frames, n_dft=512, n_hop=256):
    model = Sequential()
    model.add(Spectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=(1, num_frames),
                          return_decibel_spectrogram=True, power_spectrogram=2.0,
                          trainable_kernel=False, name='static_stft'))
    time_frames = math.ceil(num_frames/n_hop)
    scale_down = 4
    trim_by = time_frames % scale_down
    model.add(Cropping2D(((0, 1), (0, trim_by))))
    mel = Input(tensor=model.output)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(mel)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # encoded = Flatten()(encoded)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # x =
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(mel, decoded)

    # encoder
    encoder = Model(mel, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder
