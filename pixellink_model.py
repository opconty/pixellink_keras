#-*- coding:utf-8 -*-
#'''
# Created on 18-10-15
#
# @Author: Greg Gao(laygin)
#'''
import os
import numpy as np
from keras import layers
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend
import keras.backend as K
from keras.layers import Conv2D, Add, Lambda
from functools import partial
from keras.models import Model
import tensorflow as tf


def upsample(x):
    return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*2, K.shape(x)[2]*2])


def _generate_layer_name(name, prefix=None):
    if prefix is None:
        return None

    return '_'.join([prefix, name])


def pixellink_vgg16(weights='imagenet',
          input_tensor=None,
          input_shape=None,
          fatness=64,
          dilation=True,
          acf=None):
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=48,
                                          data_format=backend.image_data_format(),
                                          require_flatten=False,
                                          weights=weights)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        # Block 1
        name_fmt = partial(_generate_layer_name, prefix='conv1')
        x = layers.Conv2D(64, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv1_1'))(img_input)
        x = layers.Conv2D(64, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv1_2'))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        name_fmt = partial(_generate_layer_name, prefix='conv2')
        x = layers.Conv2D(128, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv2_1'))(x)
        x = layers.Conv2D(128, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv2_2'))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        name_fmt = partial(_generate_layer_name, prefix='conv3')
        x = layers.Conv2D(256, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv3_1'))(x)
        x = layers.Conv2D(256, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv3_2'))(x)
        x = layers.Conv2D(256, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv3_3'))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        name_fmt = partial(_generate_layer_name, prefix='conv4')
        x = layers.Conv2D(512, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv4_1'))(x)
        x = layers.Conv2D(512, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv4_2'))(x)
        x = layers.Conv2D(512, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv4_3'))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        name_fmt = partial(_generate_layer_name, prefix='conv5')
        x = layers.Conv2D(512, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv5_1'))(x)
        x = layers.Conv2D(512, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv5_2'))(x)
        x = layers.Conv2D(512, (3, 3),
                          activation=acf,
                          padding='same',
                          name=name_fmt('conv5_3'))(x)
        x = layers.MaxPooling2D((3, 3), strides=(1, 1), name='block5_pool', padding='same')(x)

        # fc layers as conv, and dilation is added
        if dilation:
            x = layers.Conv2D(fatness * 16, kernel_size=3, dilation_rate=6, padding='same', name='fc6')(x)
        else:
            x = layers.Conv2D(fatness * 16, kernel_size=3, padding='same', name='fc6')(x)

        x = layers.Conv2D(fatness * 16, kernel_size=1, padding='same', name='fc7')(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='pixellink_vgg16')

        return model


def _score_feats(filters, x, name):
    x = Conv2D(filters, 1, name=name)(x)
    return x


def _fuse_feats(filters, x1, x2, up=True):
    if filters == 2:
        name1 = 'pixel_cls_{}'.format(x1.name.split('/')[0].split('_', 1)[-1])
        name2 = 'pixel_cls_{}'.format(x2.name.split('/')[0].split('_', 1)[-1])
    else:
        name1 = 'pixel_link_{}'.format(x1.name.split('/')[0].split('_', 1)[-1])
        name2 = 'pixel_link_{}'.format(x2.name.split('/')[0].split('_', 1)[-1])

    x1 = _score_feats(filters, x1, name1)
    if up:
        x2 = Lambda(upsample)(x2)
    else:
        x2 = _score_feats(filters, x2, name2)

    return Add()([x1, x2])


def create_pixellink_model(input_shape=None, acf='relu'):
    backbone = pixellink_vgg16(input_shape=input_shape, acf=acf)
    fc7 = backbone.get_layer('fc7').output
    conv5_3 = backbone.get_layer('conv5_conv5_3').output
    conv4_3 = backbone.get_layer('conv4_conv4_3').output
    conv3_3 = backbone.get_layer('conv3_conv3_3').output

    fc7_conv5_3_cls = _fuse_feats(2, conv5_3, fc7, up=False)
    fc7_conv5_3_link = _fuse_feats(16, conv5_3, fc7, up=False)

    conv5_conv4_cls = _fuse_feats(2, conv4_3, fc7_conv5_3_cls)
    conv5_conv4_link = _fuse_feats(16, conv4_3, fc7_conv5_3_link)

    conv4_conv3_cls = _fuse_feats(2, conv3_3, conv5_conv4_cls)
    conv4_conv3_link = _fuse_feats(16, conv3_3, conv5_conv4_link)


    model = Model(backbone.input, [conv4_conv3_cls, conv4_conv3_link], name='pixellink')

    return model