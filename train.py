import tensorflow as tf

import model

tf.app.flags.DEFINE_float("dropout", 0.8, "Dropout after the bottleneck layer")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size")

tf.app.flags.DEFINE_string("logdir", './model_files/', "Batch size")

FLAGS = tf.app.flags.FLAGS


def train(args):
    nn = model.construct_model()

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': []},
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True,
        queue_capacity=10000,
        num_threads=1,
    )
    summary_hook = tf.train.SummarySaverHook(
        save_secs=2,
        output_dir=FLAGS.logdir,
        summary_op=tf.summary.merge_all(),
        scaffold=tf.train.Scaffold()
    )
    # Train
    nn.train(input_fn=train_input_fn, steps=30000, hooks=[summary_hook])

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': piv.as_matrix()},  # TODO test data
        num_epochs=1,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])


if __name__ == "__main__":
    tf.app.run(main=train)
