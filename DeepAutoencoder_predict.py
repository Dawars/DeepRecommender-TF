import tensorflow as tf
import pandas as pd
import numpy as np

from DeepAutoencoder import model_fn




nn = tf.estimator.Estimator(model_fn=model_fn, params=None, model_dir='./deep_ae_ckpt')

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test.as_matrix()},
    num_epochs=1,
    shuffle=False)