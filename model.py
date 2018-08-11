import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def construct_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Set model params
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "dropout": FLAGS.dropout,
        "optimizer": FLAGS.optimizer
    }

    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir=LOGDIR,
                                config=tf.estimator.RunConfig(session_config=config))
    return nn

def model_fn(features, labels, mode, params):
    """

    :param features:
    :param labels:
    :param mode:
    :param params: activation function, optimizer, dropout, learning rate, shape
    :return:
    """
    # Model
    x = features['x']

    activation = params['activation']
    shape = params['shape']

    encoder1 = tf.layers.dense(inputs=x, units=shape[0], activation=activation, name='encoder1')
    encoder2 = tf.layers.dense(inputs=encoder1, units=shape[1], activation=activation, name='encoder2')

    bottleneck = tf.layers.dense(inputs=encoder2, units=shape[2], name='bottleneck')
    dropout = tf.layers.dropout(inputs=bottleneck, rate=params['dropout'],
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder1 = tf.layers.dense(inputs=dropout, units=shape[3], activation=activation, name='decoder1')
    decoder2 = tf.layers.dense(inputs=decoder1, units=shape[4], activation=activation, name='decoder2')

    y = tf.layers.dense(inputs=decoder2, units=x.get_shape()[1], activation=None, name="out")

    # Dense re-feeding in the same pass (not two optimization step per iteration)
    encoder1_re = tf.layers.dense(inputs=y, units=shape[0], activation=activation, reuse=True, name='encoder1')
    encoder2_re = tf.layers.dense(inputs=encoder1_re, units=shape[1], activation=activation, reuse=True, name='encoder2')

    bottleneck_re = tf.layers.dense(inputs=encoder2_re, units=shape[2], reuse=True, name='bottleneck')
    dropout_re = tf.layers.dropout(inputs=bottleneck_re, rate=params['dropout'],
                                   training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder1_re = tf.layers.dense(inputs=dropout_re, units=shape[3], activation=activation, reuse=True, name='decoder1')
    decoder2_re = tf.layers.dense(inputs=decoder1_re, units=shape[4], activation=activation, reuse=True, name='decoder2')

    y_re = tf.layers.dense(inputs=decoder2_re, units=x.get_shape()[1], reuse=True, activation=None, name="out")

    # Loss
    # Masked Mean Square Error
    weight = tf.cast(tf.greater(x, 0), tf.float32)
    # loss for first backprop + loss for dense re-feeding
    loss = tf.losses.mean_squared_error(x, y, weight) + tf.losses.mean_squared_error(y, y_re)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("rmse", tf.sqrt(loss))

    eval_metric_ops = {
        "loss": tf.metrics.mean_squared_error(x, y, weight),
        "rmse": tf.metrics.root_mean_squared_error(x, y, weight)
    }

    # Optimizer
    optimizer = params['optimizer'](learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, y, loss, train_op, eval_metric_ops)
