import numpy as np
import tensorflow as tf


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f[ 'x_train' ], f[ 'y_train' ]
        x_test, y_test = f[ 'x_test' ], f[ 'y_test' ]
    return (x_train, y_train), (x_test, y_test)


def data_transform(x):
    return np.array(x, np.float32) / 255.0


def load_tf_data(x, y, batch_size):
    train_data = tf.data.Dataset.from_tensor_slices((x, y))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    return train_data
