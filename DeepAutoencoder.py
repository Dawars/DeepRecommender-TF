# coding: utf-8

# # Training Deep AutoEncoders for Collaborative Filtering
# https://arxiv.org/pdf/1708.01715.pdf

# In[46]:


import argparse
import sys

import tensorflow as tf

import pandas as pd
import numpy as np
from tensorflow.python.estimator.model_fn import EstimatorSpec

anime = pd.read_csv('./mangaki-data-challenge-0908/titles.csv')
ratings = pd.read_csv('./mangaki-data-challenge-0908/watched.csv')
ratings.head()

# In[49]:


ratings = ratings.replace({'dislike': '1'}, regex=True)
ratings = ratings.replace({'neutral': '2'}, regex=True)
ratings = ratings.replace({'like': '3'}, regex=True)
ratings = ratings.replace({'love': '4'}, regex=True)

ratings['rating'] = ratings['rating'].astype(float)
ratings['rating'] = ratings['rating'] / 4.0

# In[50]:


ratings.head()

# In[51]:

# Adding 0 ratings for non rated works
dummy_ratings = []

for work_id in range(len(anime)):
    # if not rated by any user
    if ratings[ratings['work_id'] == work_id].empty:
        dummy_ratings.append({'work_id': int(work_id), 'rating': 0, 'user_id': 0})

ratings = ratings.append(pd.DataFrame(dummy_ratings), ignore_index=True)

piv = ratings.pivot_table(index=['user_id'], columns=['work_id'], values='rating')
piv.fillna(0, inplace=True)
piv.head()

# In[52]:


# getting row manually
piv.loc[2]

# ## Building Model

# In[53]:


FLAGS = None

# In[54]:


tf.logging.set_verbosity(tf.logging.INFO)


# In[61]:


def model_fn(features, labels, mode, params):
    # 1. Configure the model via TensorFlow operations
    # Input Layer
    input_layer = features['x']

    encoder1 = tf.layers.dense(inputs=input_layer, units=64, activation=tf.nn.selu, name='encoder1')
    encoder2 = tf.layers.dense(inputs=encoder1, units=128, activation=tf.nn.selu, name='encoder2')

    bottleneck = tf.layers.dense(inputs=encoder2, units=128, name='bottleneck')
    dropout = tf.layers.dropout(inputs=bottleneck, rate=params['dropout'], training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder1 = tf.layers.dense(inputs=dropout, units=128, activation=tf.nn.selu, name='decoder1')
    decoder2 = tf.layers.dense(inputs=decoder1, units=64, activation=tf.nn.selu, name='decoder2')
    # print('shape', input_layer.get_shape)
    output_layer = tf.layers.dense(inputs=decoder2, units=input_layer.get_shape()[1], activation=None)

    predictions = output_layer

    # 2. Define the loss function for training/evaluation
    # Masked Mean Square Error 
    # mask = tf.where(input_layer != 0, 1, 0, name='loss_mask')
    # loss = tf.divide(tf.multiply(mask, tf.squared_difference(input_layer, predictions)), tf.sum(mask))

    weight = tf.cast(tf.greater(input_layer, 0), tf.float32)
    loss = tf.losses.mean_squared_error(input_layer, predictions, weight)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("rmse", tf.sqrt(loss))

    # TODO Dense re-feeding

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(input_layer, predictions, weight)
    }
    # 3. Define the training operation/optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)


# In[62]:


def main(unused_argv):
    # Load datasets
    global piv

    # Set model params
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "dropout": FLAGS.dropout
    }

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir='./model_files/deep_ae')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': piv.as_matrix()},
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True,
        queue_capacity=10000,
        num_threads=1,
    )
    summary_hook = tf.train.SummarySaverHook(
        save_secs=2,
        output_dir='./model_files/deep_ae',
        summary_op=tf.summary.merge_all(),
        scaffold=tf.train.Scaffold()
    )
    # Train
    #nn.train(input_fn=train_input_fn, steps=30000, hooks=[summary_hook])  #

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': piv.as_matrix()},  # TODO test data
        num_epochs=1,
        shuffle=False)

    # ev = nn.evaluate(input_fn=test_input_fn)
    # print("Loss: %s" % ev["loss"])
    # print("Root Mean Squared Error: %s" % ev["rmse"])


    # predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": piv.as_matrix()},
        num_epochs=1,
        shuffle=False)

    predictions_gen = nn.predict(input_fn=predict_input_fn)

    index = piv.index.values
    reverse_index = {}

    for i, j in enumerate(index):
        reverse_index[j] = i

    predictions = list(predictions_gen)

    # print(predictions)# writing output
    test = pd.read_csv('./mangaki-data-challenge-0908/test.csv')

    for i, row in test.iterrows():
        row_num = reverse_index[int(row['user_id'])]
        work_id = int(row['work_id'])
        print('user_id', row['user_id'], 'index', row_num, 'work', work_id)
        print(predictions[row_num])
        print(predictions[row_num][work_id])

    test['out'] = test.apply(lambda row: predictions[reverse_index[int(row['user_id'])]][int(row['work_id'])], axis=1)
    test.to_csv('./deep_ae1_out.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # parser.add_argument("--train_data", type=str, default="", helpful="Path to the training data.")
    # parser.add_argument("--test_data", type=str, default="", help="Path to the test data.")
    # parser.add_argument("--predict_data", type=str, default="", help="Path to the prediction data.")

    parser.add_argument("--dropout", type=float, default=0.65, help="Dropout after the bottleneck layer.")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Learning rate.")

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# In[ ]:
