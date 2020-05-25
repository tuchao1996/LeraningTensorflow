#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @desc:
 @author: ZhangTuchao
 @contact: ztchao1996@163.com
 @github: https://github.com/tuchao1996
 @time: 2020-05-25
"""

import sys, os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath('file')))
sys.path.append(base_dir)
print('sys path: {}'.format(sys.path))

from base.base_model import BaseModel
from tensorflow.keras import Model, layers
import tensorflow as tf


class TemplateModel(Model, BaseModel):
    def __init__(self, config):
        super().__init__()
        super().__init__(config)
        num_classes = config['num_classes']

        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

        self.build_model()
        self.init_saver()

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass
