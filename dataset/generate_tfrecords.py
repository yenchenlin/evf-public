import tensorflow as tf
import glob
import os
import numpy as np
from skimage.io import imread
from video_prediction import datasets


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_tf_record(output_fname, sequences):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for (images, metadata, push_name, shape_ids) in sequences:
            assert len(images) == len(metadata)
            num_frames = len(images)
            encoded_images = [image.tostring() for image in images]
            states = np.concatenate((metadata[:, :2], np.zeros((num_frames, 1))), axis=1).reshape(-1)
            actions = np.repeat(metadata[:-1, 2], 4)
            shape_ids = np.array(shape_ids)
            features = tf.train.Features(feature={
                'images/encoded': _bytes_list_feature(encoded_images),
                'actions': _floats_list_feature(actions),
                'states': _floats_list_feature(states),
                'sequence_length': _int64_feature(num_frames),
                'push_name': _bytes_feature(push_name),
                'shape_ids': _floats_list_feature(shape_ids)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


cwd = os.getcwd()
dataset_dir = os.path.join(cwd, 'dataset/omnipush/')
output_dir = os.path.join(cwd, 'dataset/omnipush-tfrecords/')
splits = ['train', 'val']
for split in splits:
    split_path = os.path.join(output_dir, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)


# Build class id mapping
shape_names = sorted(os.listdir(os.path.join(dataset_dir, 'train')))
mapping = {}
for class_id, shape_name in enumerate(shape_names):
    mapping[shape_name] = class_id

n_trajs = 0
splits = ['train', 'test']
for split in splits:
    shape_names = os.listdir(os.path.join(dataset_dir, '{}/'.format(split)))
    for shape_name in shape_names:
        sequences = []
        dnames = glob.glob(os.path.join(dataset_dir, '{}/{}/**/'.format(split, shape_name)))
        for dname in dnames:
            n_images = len(glob.glob(os.path.join(dname, '*.png')))
            images = []
            for i in range(n_images):
                fname = os.path.join(dname, '{}.png'.format(i))
                images.append(imread(fname))
            metadata = np.load(os.path.join(dname, 'actions.npy'))
            push_name = str.encode(dname.split('/')[-2])
            shape_ids = [mapping[shape_name]] * n_images
            sequences.append((images, metadata, push_name, shape_ids))
            n_trajs += 1
        output_split = 'val' if split == 'test' else 'train'
        save_tf_record(os.path.join(cwd, 'dataset/omnipush-tfrecords/{}/{}.tfrecord'.format(output_split, shape_name)), sequences)
        
print("tfrecords successfully generated, {} trajectories in total.".format(n_trajs))
