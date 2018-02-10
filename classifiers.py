from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Cropping2D, Reshape
from keras.models import Model, Sequential
from keras import regularizers


def genre_classifier_conv(inpdimx, inpdimy, num_genres,num_encoder=12):

    inp = Input(shape=(inpdimx, inpdimy, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1')(inp)
    x = MaxPooling2D((2, 2), padding='same', name='pool_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='ppol_2')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = Flatten(name='flat_1')(x)
    x = Dense(num_genres, activation='softmax', name='dense_1')(x)
    encoded = Dense(num_encoder, activation='softmax', name='dense_2')(x)

    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    encoder = Model(inp,encoded)

    return model, encoder
