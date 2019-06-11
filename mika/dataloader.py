import numpy as np
import tensorflow as tf
import os

class Parser(object):
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        if not data_dir.endswith('/'):
            self.data_dir += '/'
        self.image_size = image_size

    def __call__(self, proto):
        keys_to_features = {'frame_path': tf.FixedLenFeature([], tf.string),
                            'control': tf.FixedLenFeature([2], tf.float32), # steering, throttle
                            'speed': tf.FixedLenFeature([1], tf.float32),
                            'sequence': tf.FixedLenFeature([], tf.int64)}
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        current_path = tf.strings.join([self.data_dir, parsed_features['frame_path']])

        # Turn your saved image string into an array
        frame = tf.read_file(current_path)
        frame = tf.image.decode_png(frame, channels=3)

        frame = tf.image.resize_images(frame, self.image_size)

        label = parsed_features['control']
        speed = parsed_features['speed']
        sequence = parsed_features['sequence']
        return (frame, speed, sequence), label

def random_image_transforms(inputs, labels):
    images, states = inputs
    return (images, states), labels

def single_train(dataset_path, config, image_size):
    preprocessor = Preprocess(config.augment)

    dataset = tf.data.TFRecordDataset(dataset_path)
    data_dir = os.path.dirname(dataset_path)
    parser_fn = Parser(data_dir, image_size)
    dataset = dataset.map(parser_fn, num_parallel_calls=config.workers)
    dataset = dataset.map(lambda x, y: ((x[0], x[1]), y))
    dataset = dataset.repeat(-1)
    dataset = dataset.shuffle(buffer_size=config.shuffle_buffer)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.map(preprocessor)
    return dataset.prefetch(1)

def single_test(dataset_path, batch_size, workers=4, image_size=(160, 320)):
    dataset = tf.data.TFRecordDataset(dataset_path)
    data_dir = os.path.dirname(dataset_path)
    parser_fn = Parser(data_dir, image_size)
    dataset = dataset.map(parser_fn, num_parallel_calls=workers)
    dataset = dataset.map(lambda x, y: ((x[0], x[1]), y))
    dataset = dataset.repeat(-1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_test)
    return dataset.prefetch(1)

def get_dataset_size(tfrecords_file):
    return sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_file))

def image_to_float_subtract_mean(image):
    return tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

class Preprocess(object):
    def __init__(self, augment):
        self.augment = augment

    def __call__(self, inputs, controls):
        images, state = inputs
        def _process_images(images, flip):

            translate_magnitude = self.augment.translate
            images = tf.cast(images, dtype=tf.float32)
            translation = tf.random_uniform(shape=(tf.shape(images)[0], 2), minval=-translate_magnitude,
                    maxval=translate_magnitude)
            images = tf.contrib.image.translate(images, translation)
            angle = tf.random.uniform((tf.shape(images)[0],), minval=-np.pi/15, maxval=np.pi/15)
            images = tf.contrib.image.rotate(images, angle)

            if hasattr(self.augment, 'hue'):
                images = tf.image.random_hue(images, self.augment.hue)
            if hasattr(self.augment, 'min_contrast'):
                images = tf.image.random_contrast(images, self.augment.min_contrast, self.augment.max_contrast)
            if hasattr(self.augment, 'min_saturation'):
                images = tf.image.random_saturation(images, self.augment.min_saturation, self.augment.max_satutation)

            flipped_image = tf.image.flip_left_right(images)
            images = tf.where(flip, flipped_image, images)

            return image_to_float_subtract_mean(images)

        N = tf.shape(images)[0]
        should_flip = tf.random_uniform(shape=[N], minval=0.0, maxval=1.0, dtype=tf.float32) < self.augment.flip_prob
        images = _process_images(images, should_flip)

        flipper = tf.constant([-1.0, 1.0], dtype=tf.float32)
        controls = tf.where(should_flip, flipper * controls, controls)
        images = images + tf.random_normal(tf.shape(images), mean=0.0, stddev=0.1)

        return (images, state), controls

def preprocess_test(inputs, controls):
    images, states = inputs
    images = tf.map_fn(image_to_float_subtract_mean, images, dtype=tf.float32)
    return (images, states), controls

