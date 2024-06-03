import os 
from keras.models import Sequential
from keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D, BatchNormalization, Activation, SpatialDropout3D, Flatten, Dense, Bidirectional, GRU, TimeDistributed

def load_model() -> Sequential: 
    model = Sequential()

    model.add(ZeroPadding3D(padding=(1, 2, 2), input_shape=(115, 54, 90, 1)))
    model.add(Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout3D(0.3))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(ZeroPadding3D(padding=(1, 2, 2)))
    model.add(Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout3D(0.3))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout3D(0.3))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'))
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'))

    model.add(Dense(63, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    model.load_weights(os.path.join('bangla_model','checkpoint'))

    return model
