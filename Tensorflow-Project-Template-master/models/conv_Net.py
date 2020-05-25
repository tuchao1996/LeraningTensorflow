#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @desc:
 @author: ZhangTuchao
 @contact: ztchao1996@163.com
 @github: https://github.com/tuchao1996
 @time: 2020-05-25
"""

from tensorflow.keras import Model, layers
import tensorflow as tf


class ConvNet(Model):
    def __init__(self, config):
        super().__init__()
        num_classes = config.num_classes
        conv1_filters = config.conv1_filters
        conv2_filters = config.conv2_filters
        fc1_units = config.fc1_units

        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(conv1_filters, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(conv2_filters, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(fc1_units)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

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

if __name__ == '__main__':
    model = ConvNet()