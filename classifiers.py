from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Cropping2D, Reshape
from keras.models import Model, Sequential
from keras import regularizers

def genre_classifier_conv(inpdimx, inpdimy, num_genres, num_encoder=12):
    inp = Input(shape=(inpdimx, inpdimy, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1')(inp)
    x = MaxPooling2D((2, 2), padding='same', name='pool_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='ppol_2')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = Flatten(name='flat_1')(x)
    encoded = Dense(num_encoder, activation='softmax', name='dense_1')(x)
    x = Dense(num_genres, activation='softmax', name='dense_2')(encoded)

    encoder = Model(inp, encoded)

    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model, encoder

def genre_classifier_conv_all_layers(inpdimx, inpdimy, num_genres):
    inp = Input(shape=(inpdimx, inpdimy, 1))

    x1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1')(inp)
    x2 = MaxPooling2D((2, 2), padding='same', name='pool_1')(x1)
    x3 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_2')(x2)
    x4 = MaxPooling2D((2, 2), padding='same', name='ppol_2')(x3)
    x5 = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_3')(x4)
    x6 = Flatten(name='flat_1')(x5)
    x7 = Dense(num_genres, activation='softmax', name='dense_2')(x6)

    encoder = Model(inp, outputs=[x1, x3, x5])

    model = Model(inp, x7)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model, encoder