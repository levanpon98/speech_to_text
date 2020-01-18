import argparse
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from utils.tf_record import *

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_META_CSV = os.path.join(_BASE_DIR, 'meta.csv')
_DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, 'tfrecords')
_DEFAULT_DATA_PATH = "D:/Kaggle/data/speech_to_text_vietnamese/train/audio/"

_DEFAULT_DURATION = 4  # seconds
_DEFAULT_SAMPLE_RATE = 16000

_DEFAULT_VAL_SIZE = 0.1

_DEFAULT_NUM_SHARDS_TRAIN = 16
_DEFAULT_NUM_SHARDS_VAL = 2

_SEED = 2020


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--meta-data-csv', type=str, dest='meta_csv',
                        default=_DEFAULT_META_CSV,
                        help='File containing audio file-paths and '
                             'corresponding labels. (default: %(default)s)')
    parser.add_argument('--data-dir', type=str, dest='data_dir',
                        default=_DEFAULT_DATA_PATH,
                        help='File containing audio file (default: %(default)s)')
    parser.add_argument('-o', '--output-dir', type=str, dest='output_dir',
                        default=_DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecord files.'
                             '(default: %(default)s)')
    parser.add_argument('--num-shards-train', type=int,
                        dest='n_shards_train',
                        help='Number of shards to divide training set '
                             'TFRecords into. Will be estimated from other '
                             'parameters if not explicitly set.')
    parser.add_argument('--num-shards-test', type=int,
                        dest='n_shards_test',
                        help='Number of shards to divide testing set '
                             'TFRecords into. Will be estimated from other '
                             'parameters if not explicitly set.')
    parser.add_argument('--num-shards-val', type=int,
                        dest='n_shards_val',
                        help='Number of shards to divide validation set '
                             'TFRecords into. Will be estimated from other '
                             'parameters if not explicitly set.')
    parser.add_argument('--duration', type=int,
                        dest='duration',
                        default=_DEFAULT_DURATION,
                        help='The duration for the resulting fixed-length '
                             'audio-data in seconds. Longer files are '
                             'truncated. Shorter files are zero-padded. '
                             '(default: %(default)s)')
    parser.add_argument('--sample-rate', type=int,
                        dest='sample_rate',
                        default=_DEFAULT_SAMPLE_RATE,
                        help='The _actual_ sample-rate of wav-files to '
                             'convert. Re-sampling is not yet supported. '
                             '(default: %(default)s)')
    parser.add_argument('--val-size', type=float,
                        dest='val_size',
                        default=_DEFAULT_VAL_SIZE,
                        help='Fraction of examples in the validation set. '
                             '(default: %(default)s)')

    return parser.parse_args()


def main(args):
    converter = TFRecordsConverter(args.meta_csv,
                                   args.output_dir,
                                   args.data_dir,
                                   args.n_shards_train,
                                   args.n_shards_val,
                                   args.duration,
                                   args.sample_rate,
                                   args.val_size)
    converter.convert()


if __name__ == '__main__':
    main(parse_args())
