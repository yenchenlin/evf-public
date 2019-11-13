from collections import OrderedDict

import numpy as np
import tensorflow as tf

# Reuse components from SAVP
from .savp_model import SAVPVideoPredictionModel

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


class EVFVideoPredictionModel(SAVPVideoPredictionModel):
    # Customized graph to introduce experience embedding encoder.
    def build_graph(self, inputs):
        batch_size = inputs['images'].shape[0]
        seq_len = inputs['images'].shape[1]

        # Use first `self.exp_size` of inputs as experience
        exp = OrderedDict()
        for key, value in inputs.items():
            exp[key] = value[:self.exp_size]
        
        # Concat first and last frames to represent a trajectory
        first_frames = inputs['images'][:, 0, :, :, :]
        last_frames = inputs['images'][:, -1, :, :, :]
        input_frames = tf.concat(axis=3, values=[first_frames, last_frames])

        # Construct an encoder
        enc1 = slim.layers.conv2d(    #32x32x32
            input_frames,
            32, [5, 5],
            stride=2,
            scope='conv1',
            normalizer_fn=tf_layers.layer_norm,
            normalizer_params={'scope': 'layer_norm1'})      
        enc2 = slim.layers.conv2d(    #16x16x64
            enc1,
            64, [3, 3],
            stride=2,
            scope='conv2',
            normalizer_fn=tf_layers.layer_norm,
            normalizer_params={'scope': 'layer_norm2'})
        enc3 = slim.layers.conv2d(    #8x8x64
            enc2,
            64, [3, 3],
            stride=2,
            scope='conv3',
            normalizer_fn=tf_layers.layer_norm,
            normalizer_params={'scope': 'layer_norm3'})
        enc4 = slim.layers.conv2d(    #4x4x64
            enc3,
            64, [3, 3],
            stride=2,
            scope='conv4',
            normalizer_fn=tf_layers.layer_norm,
            normalizer_params={'scope': 'layer_norm4'})

        # Get embedding from experience before multi-gpu split
        enc4_flat = tf.reshape(enc4, [int(batch_size), -1])
        embedding = slim.layers.fully_connected(
                        enc4_flat,
                        5,
                        scope='embedding',
                        activation_fn=None)
        embedding = tf.tile(
            tf.reshape(embedding, [int(batch_size), 1, -1]), [1, seq_len-1, 1]
        )

        # Add the embedding back to inputs
        inputs['embeddings'] = embedding

        super(EVFVideoPredictionModel, self).build_graph(inputs)