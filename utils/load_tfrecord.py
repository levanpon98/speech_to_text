import os
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


def _parse_batch(record_batch):
    n_sample = 20 * 200

    feature_description = {
        'audio': tf.io.FixedLenFeature([n_sample], tf.float32),
        'label': tf.io.VarLenFeature(tf.int64)
    }

    example = tf.io.parse_example(record_batch, feature_description)

    return example['audio'], example['label']
    # return example['audio'], example['label']


def get_dataset_from_tfrecords(tfrecords_dir='tfrecords', split='train', batch_size=16,
                               sample_rate=44100, duration=4, n_epochs=10):
    if split not in ('train', 'validate'):
        raise ValueError("Split must be either 'train' or 'validate'")

    pattern = os.path.join(tfrecords_dir, '{}*.tfrecord'.format(split))

    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB')
    # Prepare batches
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x), num_parallel_calls=AUTO)
    # ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration), num_parallel_calls=AUTO)

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    if split == 'train':
        ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTO)
    # return ds
