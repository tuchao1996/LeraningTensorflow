import tensorflow as tf


def loss_func(x, y):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.cast(y, tf.int64), x)
    loss = tf.reduce_mean(loss)
    return loss


def optimizer_func(learning_rate):
    return tf.optimizers.Adam(learning_rate)


def metrics_func(y_pred, y_true, metrics_type):
    res = []
    if 'acc' in metrics_type:
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), -1)
        res.append(acc.numpy())
    return res


def train_step(model, x, y, learning_rate):
    with tf.GradientTape() as g:
        y_pred = model(x, is_training=True)
        loss = loss_func(y_pred, y)
        gradients = g.gradient(target=loss, sources=model.trainable_variables)
        optimizer = optimizer_func(learning_rate)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))


def train_epoch(train_data, model, learning_rate, training_steps, display_steps, metrics_type):
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        train_step(model, batch_x, batch_y, learning_rate)
        if not step % display_steps:
            pred = model(batch_x)
            loss = loss_func(pred, batch_y)
            acc = metrics_func(pred, batch_y, metrics_type)
            print('step:{} loss:{} accuracy:{}'.format(step, loss, acc))
    return model