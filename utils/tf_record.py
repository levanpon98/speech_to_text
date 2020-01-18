import math
import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.text import *
from tqdm import tqdm
import librosa

_SEED = 2020

pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def _parallelize(func, data):
    processes = cpu_count() - 1
    with Pool(processes) as pool:
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))


class TFRecordsConverter:
    """Convert WAV files to TFRecords."""

    # When compression is used, resulting TFRecord files are four to five times
    # smaller. So, we can reduce the number of shards by this factor
    _COMPRESSION_SCALING_FACTOR = 4

    def __init__(self, meta, output_dir, data_dir, n_shards_train,
                 n_shards_val, duration, sample_rate, val_size):
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.duration = duration
        self.sample_rate = sample_rate

        df = pd.read_csv(meta, index_col=False)
        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=_SEED)

        n_samples = len(df)
        self.n_val = math.ceil(n_samples * val_size)
        self.n_train = n_samples - self.n_val

        if n_shards_train is None or n_shards_val is None:
            self.n_shards_train = self._n_shards(self.n_train)
            self.n_shards_val = self._n_shards(self.n_val)
        else:
            self.n_shards_train = n_shards_train
            self.n_shards_val = n_shards_val

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __repr__(self):
        return ('{}.{}(output_dir={}, n_shards_train={}, '
                'n_shards_val={}, duration={}, sample_rate={}, n_train={}, '
                'n_test={}, n_val={})').format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.output_dir,
            self.n_shards_train,
            self.n_shards_val,
            self.duration,
            self.sample_rate,
            self.n_train,
            self.n_val,
        )

    def _n_shards(self, n_samples):
        """Compute number of shards for number of samples.
        TFRecords are split into multiple shards. Each shard's size should be
        between 100 MB and 200 MB according to the TensorFlow documentation.
        Parameters
        ----------
        n_samples : int
            The number of samples to split into TFRecord shards.
        Returns
        -------
        n_shards : int
            The number of shards needed to fit the provided number of samples.
        """
        return math.ceil(n_samples / self._shard_size())

    def _shard_size(self):
        """Compute the shard size.
        Computes how many WAV files with the given sample-rate and duration
        fit into one TFRecord shard to stay within the 100 MB - 200 MB limit.
        Returns
        -------
        shard_size : int
            The number samples one shard can contain.
        """
        shard_max_bytes = 200 * 1024 ** 2  # 200 MB maximum
        audio_bytes_per_second = self.sample_rate * 2  # 16-bit audio
        audio_bytes_total = audio_bytes_per_second * self.duration
        shard_size = shard_max_bytes // audio_bytes_total
        return shard_size * self._COMPRESSION_SCALING_FACTOR

    def _write_tfrecord_file(self, shard_data):
        """Write TFRecord file.
        Parameters
        ----------
        shard_data : tuple (str, list)
            A tuple containing the shard path and the list of indices to write
            to it.
        """
        shard_path, indices = shard_data
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for index in indices:
                file_path = self.data_dir + self.df['Filename'][index] + ".wav"
                label = str2index(self.df['Text'][index])

                audio, sample_rate = librosa.core.load(file_path, sr=self.sample_rate)
                # audio = stretch(audio, sr=self.sample_rate, duration=self.duration)
                audio = librosa.feature.mfcc(audio)
                audio = pad2d(audio, 200)
                # raw_audio = tf.io.read_file(file_path)
                # audio, sample_rate = tf.audio.decode_wav(
                #     raw_audio,
                #     desired_channels=1,  # mono
                #     desired_samples=self.sample_rate * self.duration)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'audio': _float_feature(audio),
                    'label': _int64_feature(label)}))

                out.write(example.SerializeToString())

    def _get_shard_path(self, split, shard_id, shard_size):
        """Construct a shard file path.
        Parameters
        ----------
        split : str
            The data split. Typically 'train', 'test' or 'validate'.
        shard_id : int
            The shard ID.
        shard_size : int
            The number of samples this shard contains.
        Returns
        -------
        shard_path : str
            The constructed shard path.
        """
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}.tfrecord'.format(split, shard_id,
                                                           shard_size))

    def _split_data_into_shards(self):
        """Split data into train/test/val sets.
        Split data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        Returns
        -------
        shards : list [tuple]
            The shards as a list of tuples. Each item in this list is a tuple
            which contains the shard path and a list of indices to write to it.
        """
        shards = []

        splits = ('train', 'validate')
        split_sizes = (self.n_train, self.n_val)
        split_n_shards = (self.n_shards_train, self.n_shards_val)

        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            print('Splitting {} set into TFRecord shards...'.format(split))
            shard_size = math.ceil(size / n_shards)
            cumulative_size = offset + size
            for shard_id in range(1, n_shards + 1):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)
                # Select a subset of indices to get only a subset of
                # audio-files/labels for the current shard.
                file_indices = np.arange(offset, offset + step_size)
                shards.append((shard_path, file_indices))
                offset += step_size

        return shards

    def convert(self):
        """Convert to TFRecords."""
        shard_splits = self._split_data_into_shards()
        _parallelize(self._write_tfrecord_file, shard_splits)

        print('Number of training examples: {}'.format(self.n_train))
        print('Number of validation examples: {}'.format(self.n_val))
        print('TFRecord files saved to {}'.format(self.output_dir))
