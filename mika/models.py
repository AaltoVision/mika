import tensorflow as tf
from tensorflow import keras


def simple_block(x_in):
    maps = x_in.shape[-1].value // 2

    concat = tf.keras.layers.Concatenate(axis=-1)

    x = tf.keras.layers.Conv2D(maps, kernel_size=1, activation=tf.nn.elu, strides=1, use_bias=False)(x_in)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x1 = tf.keras.layers.Conv2D(maps, kernel_size=3, activation=tf.nn.elu, padding="same")(x)
    x2 = tf.keras.layers.Conv2D(maps, kernel_size=3, activation=tf.nn.elu, padding="same")(concat([x, x1]))
    return tf.keras.layers.Conv2D(maps * 2, kernel_size=3, activation=tf.nn.elu, padding="same")(concat([x, x1, x2]))

def bc_model(frame, state, activation='elu', dropout=0.5):
    model_output = keras.layers.Conv2D(16, (7, 7), activation=activation, strides=2)(frame)
    model_output = keras.layers.BatchNormalization(axis=-1)(model_output)
    model_output = keras.layers.Conv2D(32, (4, 4), activation=activation, strides=2)(model_output)
    model_output = simple_block(model_output)
    model_output = keras.layers.MaxPool2D(pool_size=2, strides=2)(model_output)
    model_output = simple_block(model_output)
    model_output = keras.layers.MaxPool2D(pool_size=2, strides=2)(model_output)
    model_output = simple_block(model_output)

    model_output = keras.layers.Flatten()(model_output)
    model_output = keras.layers.Concatenate(axis=-1)([model_output, state])
    model_output = keras.layers.Dropout(dropout)(model_output)
    model_output = keras.layers.Dense(100, activation=activation)(model_output)
    model_output = keras.layers.Dense(50, activation=activation)(model_output)
    model_output = keras.layers.Dense(10, activation=activation)(model_output)
    model_output = keras.layers.Dense(2, activation=tf.nn.tanh, name='control')(model_output)

    return keras.models.Model(inputs=[frame, state], outputs=model_output)

