import os
import sys
import tensorflow as tf
from tensorflow import keras
import argparse
import models
import dataloader
from config import Config, Argument as arg

IMG_WIDTH = 320
IMG_HEIGHT = 180
D_STATE = 1

config = Config(
        dataset=arg(str, 'dataset', 'TFRecords file'),
        validation=arg(str, 'validation', 'TFRecords file'),
        model_name=arg(str, 'bc', 'Output directory'),
        batch_size=arg(int, 64, "Batch size"),
        epochs=arg(int, 50, 'How many epochs to train'),
        lr=arg(float, 1e-4, 'Learning rate'),
        shuffle_buffer=arg(int, 8000, 'Shuffle buffer size'),
        workers=arg(int, 4, 'How many threads to use for preprocessing'),
        augment=Config(
            flip_prob=arg(float, 0.2, 'Probability of flipping image in preprocessing'),
            translate=arg(int, 25, 'translation magnitude in preprocessing'),
            min_contrast=arg(float, 0.9, 'Min contrast augmentation'),
            max_contrast=arg(float, 1.1, 'Max contrast augmentation'),
            min_saturation=arg(float, 0.9, 'Min saturation'),
            max_satutation=arg(float, 1.1, 'Max saturation'),
            hue=arg(float, 0.1, 'Hue max delta')
            )
        )

def main():
    config.parse_arguments(sys.argv)

    image = keras.Input([IMG_HEIGHT, IMG_WIDTH, 3], name='frame')
    state = keras.Input([D_STATE], name='state')
    train_model = models.bc_model(image, state)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, verbose=1,
                                                  mode='auto', patience=10,
                                                  monitor='loss', min_lr=1e-6)
    train_model.compile(loss={'control': 'mean_squared_error'},
                        optimizer=tf.keras.optimizers.Adam(lr=config.lr))

    train_model.summary()
    ds_train = dataloader.single_train(config.dataset, config, image_size=(IMG_HEIGHT, IMG_WIDTH))
    ds_test = dataloader.single_test(config.validation, 128, image_size=(IMG_HEIGHT, IMG_WIDTH))

    model_path = os.path.join('models', config.model_name)
    os.makedirs(model_path, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'model_weights.h5'),
                                                    monitor='loss',
                                                    verbose=0,
                                                    save_best_only=False,
                                                    mode='auto')

    TRAIN_SIZE = dataloader.get_dataset_size(config.dataset)
    steps_per_epoch = TRAIN_SIZE // config.batch_size
    validation_step = dataloader.get_dataset_size(config.validation) // config.batch_size

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=model_path)
    train_model.fit(ds_train, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
            validation_data=ds_test,
            validation_steps=validation_step,
            callbacks=[checkpoint, tensorboard, reduce_lr])

    return 0


if __name__ == '__main__':
    main()

