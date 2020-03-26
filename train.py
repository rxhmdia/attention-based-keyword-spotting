import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
from model import GRUModel
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
    model = GRUModel(params.feature_size, params.gru_num_units,
                     params.attention_size)
    if (params.window_size == 1):
        logits = model.build_full_model_without_attention(
            features, params.feature_extract_layers)
    else:
        logits = model.build_full_model_with_window(
            features, params.feature_extract_layers, params.window_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.one_hot(labels, 2)
        epsilon = 1e-8
        if params.focal_loss:
            loss = tf.reduce_mean(-tf.reduce_sum(tf.pow(1 - logits, 2) *
                                                 labels *
                                                 tf.log(logits + epsilon),
                                                 reduction_indices=[1]))
        else:
            loss = tf.reduce_mean(-tf.reduce_sum(
                labels * tf.log(logits + epsilon), reduction_indices=[1]))

        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(learning_rate,
                                              global_step,
                                              decay_steps=2000,
                                              decay_rate=0.9)

        tv = tf.trainable_variables()
        regularization_cost = 5e-4 * tf.reduce_sum(
            [tf.nn.l2_loss(v) for v in tv])
        tf.summary.scalar("regularization_cost", regularization_cost)
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss + regularization_cost,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=optimizer,
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': logits}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        raise NotImplementedError()


def calc_mean_variance(CMVN_json):
    '''
    cmvn.txt contains three lines, first line is '[' that should be ommitted
    second line is a vector whose size is feature_dim + 1, it's the sum of
    feature values in each dimension, the last value is the total number of
    frames;
    the third line is the sum of sequares of feature values in each dimension,
    vector size is also feature_dim + 1
    '''
    cmvn_stats_path = CMVN_json
    with open(cmvn_stats_path) as f:
        _ = f.readline()
        val_sum = f.readline()
        square_sum = f.readline()
    val_sum = val_sum.strip().split(' ')
    n = float(val_sum.pop())
    mean = np.array([float(s) for s in val_sum]) / n
    variance = np.array([float(s) for s in square_sum.strip().split(' ')[:23]
                         ]) / n - np.square(mean)
    return mean, variance


def get_CMVN(CMVN_json):
    with open(CMVN_json) as f:
        stat = json.load(f)
        mean = np.array(stat['mean'])
        scale = np.array(stat['scale'])
    return mean, scale


def splice(nnet_input, left_context, right_context):
    res = []
    num_rows = tf.shape(nnet_input)[0]
    first_frame = tf.slice(nnet_input, [0, 0], [1, -1])
    last_frame = tf.slice(nnet_input, [num_rows - 1, 0], [1, -1])
    left_padding = tf.tile(first_frame, [left_context, 1])
    right_padding = tf.tile(last_frame, [right_context, 1])
    padded_input = tf.concat([left_padding, nnet_input, right_padding], 0)
    for i in range(left_context + right_context + 1):
        frame = tf.slice(padded_input, [i, 0], [num_rows, -1])
        res.append(frame)
    return tf.concat(res, 1)


def input_fn(params):
    data_tfrecords = []
    data_labels = []
    if not params.CMVN_json == None:
        mean, scale = get_CMVN(params.CMVN_json)
    for i in open(params.train_data_path, 'r'):
        i = i.strip()
        tmp = i.split(' ')
        data_tfrecords.append(tmp[0])
        data_labels.append(int(tmp[1]))
    features = tf.data.TFRecordDataset(data_tfrecords)
    labels = tf.data.Dataset.from_tensor_slices(data_labels)

    def parse(proto):
        features = {
            "nnet_input":
            tf.FixedLenSequenceFeature(shape=[params.feature_size],
                                       dtype=tf.float32),
        }
        _, example = tf.parse_single_sequence_example(
            proto, sequence_features=features)
        nnet_input = tf.reshape(example['nnet_input'],
                                [-1, params.feature_size])
        if not params.CMVN_json == None:
            nnet_input = (nnet_input - tf.convert_to_tensor(
                mean, dtype=tf.float32)) * tf.convert_to_tensor(
                    scale, dtype=tf.float32)
        if params.left_context > 0 or params.right_context > 0:
            nnet_input = splice(nnet_input, params.left_context,
                                params.right_context)
        return nnet_input

    dataset = features.map(parse, num_parallel_calls=32)
    data_with_label = (tf.data.Dataset.zip(
        (dataset, labels))).prefetch(params.shuffle_size)
    data_with_label = data_with_label.shuffle(buffer_size=params.shuffle_size)
    data_with_label = data_with_label.repeat(1)
    data_with_label = data_with_label.padded_batch(
        params.batch_size,
        padded_shapes=([
            None, params.feature_size *
            (params.left_context + params.right_context + 1)
        ], [])).prefetch(1)
    #iterator = data_with_label.make_one_shot_iterator()
    #data, labels = iterator.get_next()
    #features = dict()
    #features['feature'] = data
    return data_with_label  #.make_one_shot_iterator().get_next() #features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data')
    parser.add_argument('--output_dir')
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--CMVN_json', default=None)
    parser.add_argument('--focal_loss', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--shuffle_size', type=int, default=100000)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--left_context', type=int, default=0)
    parser.add_argument('--right_context', type=int, default=0)
    parser.add_argument('--feature_extract_layers', nargs='+', type=int)
    args = parser.parse_args()
    # Setup estimator config
    #NUM_GPUS=2
    #strategy=tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
    session_config = tf.ConfigProto()
    #session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(
        session_config=session_config, keep_checkpoint_max=10)
    # Setup hyper parameters
    hparams = HParams(CMVN_json=args.CMVN_json,
                      train_data_path=args.train_data,
                      learning_rate=args.learning_rate,
                      shuffle_size=args.shuffle_size,
                      max_epochs=args.max_epochs,
                      batch_size=args.batch_size,
                      feature_size=23,
                      gru_num_units=128,
                      attention_size=128,
                      clip_gradients=5.0,
                      left_context=args.left_context,
                      right_context=args.right_context,
                      focal_loss=args.focal_loss,
                      window_size=args.window_size,
                      feature_extract_layers=args.feature_extract_layers)
    # Create estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=args.output_dir,
                                       config=run_config,
                                       params=hparams)
    # Run training
    train_tensors_to_log = {"loss": "loss"}
    train_logging_hook = [
        tf.train.LoggingTensorHook(tensors=train_tensors_to_log,
                                   every_n_iter=200)
    ]
    for _ in range(args.max_epochs):
        estimator.train(input_fn=lambda: input_fn(hparams),
                        hooks=train_logging_hook)
