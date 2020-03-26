import matplotlib
matplotlib.use('Agg')
import json
import argparse
import sys
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt,mpld3
from model_streaming import GRUModel
np.set_printoptions(threshold=np.nan)


def get_CMVN(CMVN_json):
    with open(CMVN_json) as f:
        stat = json.load(f)
        mean = np.array(stat['mean'])
        scale = np.array(stat['scale'])
    return mean, scale


def input_fn(params):
    data_tfrecords = []
    if not (params.CMVN_json == None):
        mean, scale = get_CMVN(params.CMVN_json)
        #mean,scale=calc_mean_variance(params.CMVN_json)
    for i in open(params.train_data_path, 'r'):
        i = i.strip()
        tmp = i.split(' ')
        data_tfrecords.append(tmp[1])
    features = tf.data.TFRecordDataset(data_tfrecords)

    def parse(proto):
        features = {
            "feature":
            tf.FixedLenSequenceFeature(shape=[params.feature_size],
                                       dtype=tf.float32),
        }
        _, example = tf.parse_single_sequence_example(
            proto, sequence_features=features)
        feature = tf.reshape(example['feature'], [-1, params.feature_size])
        if not (params.CMVN_json == None):
            feature = (feature - tf.convert_to_tensor(mean, dtype=tf.float32)
                       ) * tf.convert_to_tensor(scale, dtype=tf.float32)
        return feature

    dataset = features.map(parse, num_parallel_calls=32)
    dataset = dataset.padded_batch(params.batch_size,
                                   padded_shapes=([None, params.feature_size
                                                   ])).prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    features = dict()
    features['feature'] = data
    return features


# get rnn output
def get_rnn_output(model, input_features, layer_norm):
    rnn_output = model.get_rnn_output(input_features, layer_norm)
    return rnn_output


# rnn output as input, get attention and the prediction
def get_attention(model, rnn_output):
    logits, alpha = model.get_attention(rnn_output)
    return tf.nn.softmax(logits), alpha


def main(_):
    test_data_path = args.test_data_path
    model_path = tf.train.latest_checkpoint(args.model_path)
    # To draw roc, it is required to set batch_size 1
    batch_size = int(args.batch_size)
    # tfrecords names
    test_data_tfrecords = []
    for i in open(test_data_path, 'r'):
        i = i.strip()
        tmp = i.split(' ')
        test_data_tfrecords.append(tmp[1])

    # params
    params = dict(train_data_path=test_data_path,
                  learning_rate=1e-2,
                  gru_num_units=128,
                  feature_size=23,
                  batch_size=batch_size,
                  attention_size=128,
                  net_type='GRU',
                  CMVN_json=args.CMVN_json,
                  layer_norm=args.layer_norm)
    hparams = tf.contrib.training.HParams(**params)

    # get input
    features_lengths = input_fn(hparams)

    # define network
    features_frame = tf.placeholder(tf.float32,
                                    shape=[None, 20, hparams.gru_num_units])
    model = GRUModel(hparams.feature_size, hparams.gru_num_units,
                     hparams.attention_size)
    rnn_output, state = get_rnn_output(model,
                                       features_lengths['feature'],
                                       layer_norm=args.layer_norm)
    predictions, attention_weight = get_attention(model, features_frame)
    # config
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, model_path)
    count = 0
    window_size = 20
    f = open(args.output_confidence, 'w')
    while True:
        try:
            batch_name = []
            batch_frames_confidence = [[] for i in range(hparams.batch_size)]
            start_val = 0
            rnn_output_val = sess.run(rnn_output)
            shape_rnn = np.array(rnn_output_val).shape
            #print(shape_rnn)
            for i in range(count * batch_size, (count + 1) * batch_size):
                if i < len(test_data_tfrecords):
                    batch_name.append((test_data_tfrecords[i].split('/')[-1]
                                       ).split('.tfrecords')[0])
            count += 1
            for i in range(shape_rnn[1] - window_size):
                feature_step = rnn_output_val[:, i:i + window_size, :]
                predictions_val, attention_weight_val = sess.run(
                    [predictions, attention_weight],
                    feed_dict={
                        features_frame: rnn_output_val[:, i:i + window_size, :]
                    })
                #print(predictions_val)
                for c in range(shape_rnn[0]):
                    batch_frames_confidence[c].append(predictions_val[c][1])
            for i in range(shape_rnn[0]):
                f.write(batch_name[i])
                f.write(' [')
                f.write('\n')
                for pred in batch_frames_confidence[i]:
                    f.write(str(pred))
                    f.write('\n')
                f.write(']')
                f.write('\n')
        except tf.errors.OutOfRangeError:
            break
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path')
    parser.add_argument('--model_path')
    parser.add_argument('--batch_size')
    parser.add_argument('--output_confidence')
    parser.add_argument('--CMVN_json', default=None)
    parser.add_argument('--layer_norm', default=False)
    args = parser.parse_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
