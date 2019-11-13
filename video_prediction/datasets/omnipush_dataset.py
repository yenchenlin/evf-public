import itertools
import os
import re
import random
import glob
from collections import OrderedDict

import tensorflow as tf


from video_prediction.utils import tf_utils
from .base_dataset import VarLenFeatureVideoDataset


class OmnipushVideoDataset(VarLenFeatureVideoDataset):
    """
    https://sites.google.com/view/sna-visual-mpc
    """
    def __init__(self, *args, **kwargs):
        super(OmnipushVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (64, 64, 3)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = 'states', (3,)
            self.action_like_names_and_shapes['actions'] = 'actions', (4,)

    def get_default_hparams_dict(self):
        default_hparams = super(OmnipushVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=24,
            time_shift=1
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parser(self, serialized_example):
        """
        Parses a single tf.train.SequenceExample into images, states, actions, etc tensors.
        """
        features = dict()
        features['sequence_length'] = tf.FixedLenFeature((), tf.int64)
        features['push_name'] = tf.VarLenFeature(tf.string)
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':
                features[name] = tf.VarLenFeature(tf.string)
            else:
                features[name] = tf.VarLenFeature(tf.float32)
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            features[name] = tf.VarLenFeature(tf.float32)

        features = tf.parse_single_example(serialized_example, features=features)

        example_sequence_length = features['sequence_length']
        state_like_seqs = OrderedDict()
        action_like_seqs = OrderedDict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':
                seq = tf.sparse_tensor_to_dense(features[name], '')
            else:
                seq = tf.sparse_tensor_to_dense(features[name])
                seq = tf.reshape(seq, [example_sequence_length] + list(shape))
            state_like_seqs[example_name] = seq
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            seq = tf.sparse_tensor_to_dense(features[name])
            seq = tf.reshape(seq, [example_sequence_length - 1] + list(shape))
            action_like_seqs[example_name] = seq

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        # decode and preprocess images on the sampled slice only
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)
        state_like_seqs['push_name'] = tf.sparse_tensor_to_dense(features['push_name'], '')
        return state_like_seqs, action_like_seqs

    @property
    def jpeg_encoding(self):
        return False

    # TODO: automatically calculate it.
    # Now manually calculate it in ipython notebook.
    def num_examples_per_epoch(self):
        return self.num_examples


class OmnipushObjectDataset(OmnipushVideoDataset):
    def __init__(self, filename, mode='train', num_epochs=None, seed=None,
                 hparams_dict=None, hparams=None, *args, **kwargs):
        """
        Dataset class for a SINGLE omnipush object.
        Args:
            filename: a tfrecord containing the data for an object.
            mode: either train, val, or test
            num_epochs: if None, dataset is iterated indefinitely.
            seed: random seed for the op that samples subsequences.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).

        Note:
            self.filename is the filename of an object's tfrecord.
        """
        self.filename = filename
        self.filenames = [filename]  # For compatibility
        if not os.path.exists(self.filename):
            raise FileNotFoundError("tfrecord %s does not exist" % self.filename)
        self.mode = mode
        self.num_epochs = num_epochs
        self.seed = seed
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)
        self.num_examples = 0
        self.num_examples += sum(1 for _ in tf.python_io.tf_record_iterator(self.filename))
        self.dataset_name = os.path.split(self.filename)[-1].strip('.tfrecord')
        self.state_like_names_and_shapes = OrderedDict()
        self.action_like_names_and_shapes = OrderedDict()

        self.hparams = self.parse_hparams(hparams_dict, hparams)

        # Copied from OmnipushVideoDataset
        self.state_like_names_and_shapes['images'] = 'images/encoded', (64, 64, 3)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = 'states', (3,)
            self.action_like_names_and_shapes['actions'] = 'actions', (4,)

    def make_dataset(self, batch_size, skip_size=0, take_size=-1):
        filenames = self.filenames
        shuffle = self.mode == 'train' # or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024).skip(skip_size).take(take_size)
        dataset = dataset.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=256, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        def _parser(serialized_example):
            state_like_seqs, action_like_seqs = self.parser(serialized_example)
            seqs = OrderedDict(list(state_like_seqs.items()) + list(action_like_seqs.items()))
            return seqs

        num_parallel_calls = None if shuffle else 1  # for reproducibility (e.g. sampled subclips from the test set)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        dataset = dataset.prefetch(batch_size)
        return dataset


def read_dataset(data_dir, mode='train', hparams_dict=None, hparams=None):
    """
    Iterate over the objects in a data directory.
    Args:
      data_dir: a directory of objects' tfrecords.
    Returns:
      An iterable over OmnipushObjectDataset. The dataset is unaugmented and not split up into
      training and test sets.
    """
    filenames = os.path.join(os.getcwd(), data_dir, mode, "*.tfrecord")
    for object_tfrecord in sorted(glob.glob(filenames)):
        if 'tfrecord' not in object_tfrecord:
            continue
        yield OmnipushObjectDataset(object_tfrecord,
                                    mode=mode,
                                    hparams_dict=hparams_dict,
                                    hparams=hparams)
