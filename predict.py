import tensorflow as tf

import model
def predict():
    nn = model.construct_model()

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": piv.as_matrix()},
        num_epochs=1,
        shuffle=False)

    predictions_gen = nn.predict(input_fn=predict_input_fn)