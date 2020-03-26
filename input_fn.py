import functools
import json
import tensorflow as tf
import numpy as np
import wave


def get_CMVN(CMVN_json):
    with open(CMVN_json) as f:
        stat = json.load(f)
        mean = np.array(stat['mean'])
        scale = np.array(stat['scale'])
    return mean, scale


def splice(feature, left_context, right_context):
    res = []
    num_rows = tf.shape(feature)[0]
    first_frame = tf.slice(feature, [0, 0], [1, -1])
    last_frame = tf.slice(feature, [num_rows - 1, 0], [1, -1])
    left_padding = tf.tile(first_frame, [left_context, 1])
    right_padding = tf.tile(last_frame, [right_context, 1])
    padded_input = tf.concat([left_padding, feature, right_padding], 0)
    for i in range(left_context + right_context + 1):
        frame = tf.slice(padded_input, [i, 0], [num_rows, -1])
        res.append(frame)
    return tf.concat(res, 1)


def input_fn(params, mode):
    if mode == "train":
        data_path = params['train_data_path']
    elif mode == "val":
        data_path = params['val_data_path']
    else:
        raise ValueError("{} is not supported.".format(mode))
    data_tfrecords = []
    data_labels = []
    if not params['CMVN_json'] == None:
        mean, scale = get_CMVN(params['CMVN_json'])
    for i in open(data_path, 'r'):
        i = i.strip()
        tmp = i.split(' ')
        data_tfrecords.append(tmp[0])
        data_labels.append(int(tmp[1]))
    features = tf.data.TFRecordDataset(data_tfrecords)
    labels = tf.data.Dataset.from_tensor_slices(data_labels)

    def parse(proto):
        features = {
            "feature":
            tf.FixedLenSequenceFeature(shape=[params['feature_size']],
                                       dtype=tf.float32),
        }
        _, example = tf.parse_single_sequence_example(
            proto, sequence_features=features)
        feature = tf.reshape(example['feature'], [-1, params['feature_size']])
        if not params['CMVN_json'] == None:
            feature = (feature - tf.convert_to_tensor(mean, dtype=tf.float32)
                       ) * tf.convert_to_tensor(scale, dtype=tf.float32)
        if params['left_context'] > 0 or params['right_context'] > 0:
            feature = splice(feature, params['left_context'],
                             params['right_context'])
        return feature

    dataset = features.map(parse, num_parallel_calls=32)
    data_with_label = (tf.data.Dataset.zip(
        (dataset, labels))).prefetch(params['shuffle_size'])
    if mode == "train":
        data_with_label = data_with_label.shuffle(
            buffer_size=params['shuffle_size'])
        data_with_label = data_with_label.repeat(params['max_epochs'])
    data_with_label = data_with_label.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            params['batch_size'],
            padded_shapes=([
                None, params['feature_size'] *
                (params['left_context'] + params['right_context'] + 1)
            ], []))).prefetch(1)
    return data_with_label


def read_wavs(file_dict):
    for item in file_dict.items():
        try:
            wav = wave.open(item[0], "rb")
        except:
            print("File not exists: ", item[0])
            continue
        num_frame = wav.getnframes()
        wav_data = []
        for _ in range(num_frame):
            str_data = wav.readframes(1)
            wav_data.append(
                int.from_bytes(str_data, byteorder='little', signed=True))
        yield wav_data


def input_fn_wav(params, mode):
    if mode == "train":
        data_path = params['train_data_path']
    elif mode == "eval":
        data_path = params['val_data_path']
    else:
        raise ValueError("{} is not supported.".format(mode))
    filelist = [line.split()[0] for line in open(data_path)]
    labellist = [line.split()[1] for line in open(data_path)]
    file_dict = dict(zip(filelist, labellist))
    dataset = tf.data.Dataset.from_generator(functools.partial(
        read_wavs, file_dict),
                                             tf.int64,
                                             output_shapes=[None])
    dataset = dataset.map(lambda x: {"feature": x}, num_parallel_calls=32)
    data_labels = [1 if params['hotword'] in x else 0 for x in labellist]
    labels = tf.data.Dataset.from_tensor_slices(data_labels)
    data_with_label = tf.data.Dataset.zip((dataset, labels))
    if mode == "train":
        data_with_label = data_with_label.apply(
            tf.contrib.data.shuffle_and_repeat(params['shuffle_size'],
                                               params['max_epochs']))
    data_with_label = data_with_label.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(params['batch_size'],
                                                        padded_shapes=({
                                                            "feature": [None]
                                                        }, []))).prefetch(1)
    return data_with_label
