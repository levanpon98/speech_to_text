import numpy as np
import soundfile
import tensorflow as tf
from absl import logging
import math
from six.moves import xrange
import random
import numbers

def compute_spectrogram_feature(samples, sample_rate, stride_ms=10.0, window_ms=20.0, max_freq=None, eps=1e-14):
    """Compute the spectrograms for the input samples(waveforms).
    More about spectrogram computation, please refer to:
    https://en.wikipedia.org/wiki/Short-time_Fourier_transform.
    """
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate.")

    if stride_ms > window_ms:
        raise ValueError("Stride size must not be greater than window size.")

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
        samples, shape=nshape, strides=nstrides)
    assert np.all(
        windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return np.transpose(specgram, (1, 0))


def compute_label_feature(text, token_to_idx):
    """
    Convert string to a list of integers.
    :param text: string
    :param token_to_idx:
    :return:
    """
    tokens = list(text.strip().lower())
    feats = [token_to_idx[token] for token in tokens]
    return feats


def normalize_audio_feature(audio):
    """
    Perform mean and variance normalization on the spectrogram feature.
    :param audio: a numpy array for the spectrogram feature.
    :return: a numpy array of the normalized spectrogram
    """
    return (audio - np.mean(audio, axis=0)) / (np.sqrt(np.var(audio, axis=0)) + 1e-6)


def preprocess_audio(audio_path, audio_config, normalize):
    """
        Load the audio file and compute spectrogram feature.
    :param audio_path: string
    :param audio_config: AudioFeaturizer
    :param normalize:
    :return:
    """
    data, _ = soundfile.read(audio_path)
    feature = compute_spectrogram_feature(data,
                                          audio_config.sample_rate,
                                          audio_config.stride_ms,
                                          audio_config.window_ms)

    if normalize:
        feature = normalize_audio_feature(feature)

    feature = np.expand_dims(feature, axis=2)
    return feature


def preprocess_data(file_path):
    """

    :param file_path: a string specifying the csv file path for a meta data.
    :return: A list of tuples (wav_filename, transcript)
    """
    logging.info("Loading metadata {}".format(file_path))

    with tf.io.gfile.GFile(file_path, mode='r') as f:
        lines = f.read().splitlines()

    lines = lines[1:]
    lines = [line.split(",", 1) for line in lines]

    return [tuple(line) for line in lines]


def batch_wise_dataset_shuffle(entries, epoch_index, sortagrad, batch_size):
    """Batch-wise shuffling of the data entries.

    Each data entry is in the format of (audio_file, file_size, transcript).
    If epoch_index is 0 and sortagrad is true, we don't perform shuffling and
    return entries in sorted file_size order. Otherwise, do batch_wise shuffling.

    Args:
      entries: a list of data entries.
      epoch_index: an integer of epoch index
      sortagrad: a boolean to control whether sorting the audio in the first
        training epoch.
      batch_size: an integer for the batch size.

    Returns:
      The shuffled data entries.
    """
    shuffled_entries = []
    if epoch_index == 0 and sortagrad:
        # No need to shuffle.
        shuffled_entries = entries
    else:
        # Shuffle entries batch-wise.
        max_buckets = int(math.floor(len(entries) / batch_size))
        total_buckets = [i for i in xrange(max_buckets)]
        random.shuffle(total_buckets)
        shuffled_entries = []
        for i in total_buckets:
            shuffled_entries.extend(entries[i * batch_size: (i + 1) * batch_size])
        # If the last batch doesn't contain enough batch_size examples,
        # just append it to the shuffled_entries.
        shuffled_entries.extend(entries[max_buckets * batch_size:])

    return shuffled_entries


def per_replica_batch_size(batch_size, num_gpus):
    if num_gpus <= 1:
        return batch_size

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. Found {} '
               'GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)
    return int(batch_size / num_gpus)


def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length):
    ctc_input_length = tf.cast(tf.multiply(
        input_length, ctc_time_steps), dtype=tf.float32)
    return tf.cast(tf.math.floordiv(
        ctc_input_length, tf.cast(max_time_steps, dtype=tf.float32)), dtype=tf.int32)


def past_stop_threshold(stop_threshold, eval_metric):
    """Return a boolean representing whether a model should be stopped.
    Args:
      stop_threshold: float, the threshold above which a model should stop
        training.
      eval_metric: float, the current value of the relevant metric to check.
    Returns:
      True if training should stop, False otherwise.
    Raises:
      ValueError: if either stop_threshold or eval_metric is not a number
    """
    if stop_threshold is None:
        return False

    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions "
                         "must be a number.")

    if eval_metric >= stop_threshold:
        tf.compat.v1.logging.info(
            "Stop threshold of {} was passed with metric value {}.".format(
                stop_threshold, eval_metric))
        return True

    return False
