import argparse
import csv
import os
import shutil
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import python_io
from PIL import Image

def _is_image(filename):
    return '.png' in filename or '.jpg' in filename or '.jpeg' in filename

def _is_csv(filename):
    return '.csv' in filename

def _load_image(image_path):
    return Image.open(image_path)

def _bytes_feature(byte_data):
    if isinstance(byte_data, str):
        byte_data = bytes(byte_data, 'utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_data]))

def _float_feature(float_data):
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_data))

def _int64_feature(int64_data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_data))

def _create_feature(item, image_path, shape):
    print(image_path)
    features = {
        'frame_path': _bytes_feature(image_path),
        'control': _float_feature(item.control),
        'speed': _float_feature([item.speed]),
        'sequence': _int64_feature([item.sequence]),
        'image_shape': _int64_feature(shape),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

def _extract_sequence_id(filepath):
    filename = os.path.split(filepath)[-1]
    _, seq_id, _ = filename.split('_')
    return int(seq_id)

class LineItem(object):
    def __init__(self, image_path, control, speed, sequence):
        self.image_path = image_path
        self.control = control
        self.speed = speed
        self.sequence = sequence

class MikaDataset(object):
    def __init__(self, data_dir):
        self._read_dirs(data_dir)
        self._data_dir = data_dir

    def _read_dirs(self, data_dir):
        directories = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.directories = [f for f in directories if os.path.isdir(f)]
        self.directories.sort()

    def _image_files(self, directory):
        files = [f for f in os.listdir(directory) if _is_image(f)]
        images = {}
        for filename in files:
            frame_id = '.'.join(filename.split('.')[:-1])
            image_path = os.path.join(directory, filename)
            image_path = os.path.relpath(image_path, self._data_dir)
            images[int(frame_id)] = image_path
        return images

    def __iter__(self):
        for data_dir in self.directories:
            images = self._image_files(data_dir)
            for image_path in images.values():
                frame_id = image_path.split('/')[1].split('.')[0]
                with open(os.path.join(data_dir, "{}.cmd".format(frame_id)), 'rt') as f:
                    line = np.array([float(i) for i in f.read().split(' ')], dtype=np.float32)
                controls =  line[1:3]
                speed = line[0]

                yield LineItem(image_path, controls, speed, self.directories.index(data_dir))

class DatasetBuilder(object):
    def __init__(self, dataset, out_dir):
        self.dataset = dataset
        self.out_dir = out_dir
        self.tfr_writer = python_io.TFRecordWriter(os.path.join(out_dir, 'dataset.tfrecords'))
        self.image_dir = 'IMG'
        self.image_dir_path = os.path.join(out_dir, self.image_dir)
        ensure_dir(self.image_dir_path)

    def build_dataset(self):
        i = 0
        for item in self.dataset:
            image_path, shape = self._write_image(item.image_path, i)
            tf_feature = _create_feature(item, image_path, shape)
            self.tfr_writer.write(tf_feature.SerializeToString())
            i += 1

    def close(self):
        self.tfr_writer.close()

    def _write_image(self, image_path, index):
        # returns path to written image file and image shape.
        original_image_path = os.path.join(self.dataset._data_dir, image_path)
        image = _load_image(original_image_path)
        shape = (image.size[1], image.size[0], 3)
        relative_path = os.path.join(self.image_dir, "{:07}.png".format(index))
        destination_path = os.path.join(self.out_dir, relative_path)
        if not os.path.exists(destination_path):
            shutil.copy2(original_image_path, destination_path)

        image_path = destination_path

        return relative_path, shape

    def _to_relative_path(self, image_path):
        image_name = os.path.split(image_path)[-1]
        return os.path.join(self.image_dir, image_name)

def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-dir', required=True, type=str)
    parser.add_argument('--to-dir', required=True, type=str)
    return parser.parse_args()

def main():
    args = read_args()
    dataset = MikaDataset(args.from_dir)
    ensure_dir(args.to_dir)
    builder = DatasetBuilder(dataset, args.to_dir)
    builder.build_dataset()
    builder.close()

if __name__ == '__main__':
    main()


