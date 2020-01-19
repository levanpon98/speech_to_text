import tensorflow as tf
from utils import fn
from utils.audio import AudioFeaturizer
from utils.text import TextFeaturizer


class DeepSpeechDataset(object):

    def __init__(self, data_config):
        self.config = data_config
        self.audio = AudioFeaturizer(
            sample_rate=self.config.audio_config.sample_rate,
            window_ms=self.config.audio_config.window_ms,
            stride_ms=self.config.audio_config.stride_ms,
        )
        self.text = TextFeaturizer(
            characters_file=self.config.characters_file_path
        )

        self.speech_labels = self.text.speech_labels
        self.entries = fn.preprocess_data(self.config.data_path + 'metadata.csv')
        self.num_feature_bins = 161  # The generated spectrogram will have 161 feature bins.


def input_fn(batch_size, deep_speech, repeat=1):
    print('starting a training cycle')
    entries = deep_speech.entries
    num_feature_bins = deep_speech.num_feature_bins
    audio_featurizer = deep_speech.audio
    feature_normalize = deep_speech.config.audio_config.normalize
    text_featurizer = deep_speech.text
    data_path = deep_speech.config.data_path

    def _gen_data():
        for audio_file, transcript in entries:
            features = fn.preprocess_audio(data_path + 'audio/' + audio_file, audio_featurizer,
                                           feature_normalize)
            labels = fn.compute_label_feature(transcript, text_featurizer.token_to_index)

            input_length = [features.shape[0]]
            label_length = [len(labels)]

            yield ({
                       "features": features,
                       "input_length": input_length,
                       "label_length": label_length
                   }, labels)

    dataset = tf.data.Dataset.from_generator(
        _gen_data,
        output_types=(
            {
                "features": tf.float32,
                "input_length": tf.int32,
                "label_length": tf.int32
            },
            tf.int32),
        output_shapes=(
            {
                "features": tf.TensorShape([None, num_feature_bins, 1]),
                "input_length": tf.TensorShape([1]),
                "label_length": tf.TensorShape([1])
            },
            tf.TensorShape([None]))
    )

    # Repeat and batch the dataset
    dataset = dataset.repeat(repeat)

    # Padding the features to its max length dimensions.
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            {
                "features": tf.TensorShape([None, num_feature_bins, 1]),
                "input_length": tf.TensorShape([1]),
                "label_length": tf.TensorShape([1])
            },
            tf.TensorShape([None]))
    )

    # Prefetch to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
