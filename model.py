import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.rnn import GRUBlockCellV2


def _attention_variables(input_size, attention_size):
    w_omega = tf.Variable(initial_value=tf.random_normal(
        [input_size, attention_size], stddev=0.1),
                          name='w_omega')
    b_omega = tf.Variable(initial_value=tf.random_normal([attention_size],
                                                         stddev=0.1),
                          name='b_omage')
    u_omega = tf.Variable(initial_value=tf.random_normal([attention_size, 1],
                                                         stddev=0.1),
                          name='u_omage')
    return w_omega, b_omega, u_omega


def _attention_layer(inputs,
                     sequences_length,
                     attention_size,
                     scope='attention'):
    hidden_size = inputs.shape[2].value
    with tf.variable_scope('attention'):
        w_omega, b_omega, u_omega = _attention_variables(
            hidden_size, attention_size)
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.squeeze(tf.tensordot(v, u_omega, axes=1, name='vu'), axis=[-1])
        shape = tf.shape(vu)
        mask = tf.sequence_mask(sequences_length, shape[1])
        score_mask_values = -np.inf
        score_mask_values = score_mask_values * tf.ones_like(vu)
        vu_mask = tf.where(mask, vu, score_mask_values)
        alphas = tf.nn.softmax(vu_mask, name='alphas')
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return output, vu_mask, alphas


def _windowsize_attention_layer(inputs,
                                sequences_length,
                                attention_size,
                                window_size,
                                scope='attention'):
    input_shape = tf.shape(inputs)
    hidden_size = inputs.shape[2].value
    with tf.variable_scope('attention'):
        w_omega, b_omega, u_omega = _attention_variables(
            hidden_size, attention_size)
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        input_slice_window = tf.map_fn(lambda i: tf.slice(
            inputs, [0, i, 0], [input_shape[0], window_size, hidden_size]),
                                       tf.range(input_shape[1] - window_size +
                                                1),
                                       dtype=tf.float32)
        vu_slice_window = tf.map_fn(lambda i: tf.slice(
            vu, [0, i, 0], [input_shape[0], window_size, 1]),
                                    tf.range(input_shape[1] - window_size + 1),
                                    dtype=tf.float32)
        vu_slice_window = tf.nn.softmax(vu_slice_window, -2)
        output = tf.reduce_sum(input_slice_window * vu_slice_window, -2)
        return output, vu, vu_slice_window


class GRUModel:
    def __init__(self, feature_size, gru_num_units, attention_size):
        self.feature_size = feature_size
        self.gru_num_units = gru_num_units
        self.gru_cell = GRUBlockCellV2(num_units=gru_num_units,
                                       name='gru_cell')
        #self.gru_cell = tf.nn.rnn_cell.DropoutWrapper(self.gru_cell, state_keep_prob=0.5)
        self.attention_size = attention_size

    def _length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def _build_full_model(self, features):
        sequence_length = self._length(features)
        x, _ = tf.nn.dynamic_rnn(self.gru_cell,
                                 features,
                                 sequence_length=sequence_length,
                                 dtype=tf.float32,
                                 scope='rnn_encoder')
        x, _, _ = _attention_layer(x, sequence_length, self.attention_size)
        logits = tf.layers.dense(x, 2, name='bottleneck')
        return logits

    def build_full_model_with_window(self,
                                     features,
                                     feature_extract_layers=[],
                                     window_size=20,
                                     states=None):
        if not (feature_extract_layers == None):
            with tf.variable_scope("feature"):
                for i, size in enumerate(feature_extract_layers):
                    name = "dense" + str(i)
                    features = tf.layers.dense(features,
                                               size,
                                               activation=tf.nn.relu,
                                               name=name)
        sequence_length = self._length(features)
        x, _ = tf.nn.dynamic_rnn(self.gru_cell,
                                 features,
                                 sequence_length=sequence_length,
                                 dtype=tf.float32,
                                 scope='rnn_encoder')
        x, _, _ = _windowsize_attention_layer(x, sequence_length,
                                              self.attention_size, window_size)
        out = tf.layers.dense(x, 2, name='bottleneck')
        out = tf.nn.softmax(out, -1)
        logits = tf.stack(
            [tf.reduce_min(out[:, :, 0], 0),
             tf.reduce_max(out[:, :, 1], 0)], 1)
        return logits

    def build_full_model_without_attention(self,
                                           features,
                                           feature_extract_layers=[]):
        if not (feature_extract_layers == None):
            with tf.variable_scope("feature"):
                for i, size in enumerate(feature_extract_layers):
                    name = "dense" + str(i)
                    features = tf.layers.dense(features,
                                               size,
                                               activation=tf.nn.relu,
                                               name=name)
        sequence_length = self._length(features)
        x, _ = tf.nn.dynamic_rnn(self.gru_cell,
                                 features,
                                 sequence_length=sequence_length,
                                 dtype=tf.float32,
                                 scope='rnn_encoder')
        out = tf.layers.dense(x, 2, name='bottleneck')
        out = tf.nn.softmax(out, -1)
        logits = tf.stack(
            [tf.reduce_min(out[:, :, 0], 1),
             tf.reduce_max(out[:, :, 1], 1)], 1)
        return logits

    def _build_encoder_model(self):
        input = tf.placeholder(tf.float32, shape=(1, 23), name='input')
        state = tf.placeholder(tf.float32,
                               shape=(1, self.gru_cell.state_size),
                               name='state')
        with tf.variable_scope('rnn_encoder'):
            output, _ = self.gru_cell(input, state, scope='gru_cell')
        with tf.variable_scope('attention'):
            w_omega, b_omage, u_omage = _attention_variables(
                self.gru_num_units, self.attention_size)
        x = tf.matmul(output, w_omega) + b_omage
        x = tf.tanh(x)
        weight = tf.matmul(x, u_omage)
        return output, weight

    def _build_classifier_model(self, window_size):
        # inputs shape is set as [feature_size, window_size]
        # to avoid inefficient transpose op when converted to tflite.
        inputs = tf.placeholder(tf.float32,
                                shape=(self.gru_num_units, window_size),
                                name='inputs')
        weights = tf.placeholder(tf.float32,
                                 shape=(1, window_size),
                                 name='weights')
        alphas = tf.nn.softmax(weights)
        alphas = tf.reshape(alphas, [window_size, 1])
        x = tf.matmul(inputs, alphas)
        x = tf.reshape(x, [1, self.gru_num_units])
        logits = tf.layers.dense(x, 2, name='bottleneck')
        return logits

    def build_training_graph(self, features, labels):
        logits = self._build_full_model(features)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels, logits, reduction=tf.losses.Reduction.MEAN)
        return loss, logits

    def build_training_graph_with_window(self, features, labels):
        logits = self._build_full_model_with_window(features)
        labels = tf.one_hot(labels, 2)
        loss = tf.reduce_mean(
            -tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))
        return loss, logits

    def build_inference_graph(self, features):
        logits = self._build_full_model(features)
        probabilities = tf.nn.softmax(logits)
        return probabilities

    def _freeze_encoder_model(self, ckpt, output_file):
        with tf.Graph().as_default(), tf.Session().as_default() as sess:
            output, weight = self._build_encoder_model()
            output = tf.identity(output, name='output')
            weight = tf.identity(weight, name='weight')
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, ckpt)
            frozen_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['output', 'weight'])
            tf.train.write_graph(frozen_graph_def,
                                 os.path.dirname(output_file),
                                 os.path.basename(output_file),
                                 as_text=False)

    def _freeze_classifier_model(self, window_size, ckpt, output_file):
        with tf.Graph().as_default(), tf.Session().as_default() as sess:
            logits = self._build_classifier_model(window_size)
            _ = tf.nn.softmax(logits, name='probabilities')
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, ckpt)
            frozen_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['probabilities'])
            tf.train.write_graph(frozen_graph_def,
                                 os.path.dirname(output_file),
                                 os.path.basename(output_file),
                                 as_text=False)

    def freeze_model(self, window_size, ckpt, output_encoder_model,
                     output_classifier_model):
        self._freeze_encoder_model(ckpt, output_encoder_model)
        self._freeze_classifier_model(window_size, ckpt,
                                      output_classifier_model)
