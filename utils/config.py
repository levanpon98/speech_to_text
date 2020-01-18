
class AudioConfig(object):
    """Configs for spectrogram extraction from audio."""

    def __init__(self,
                 sample_rate,
                 window_ms,
                 stride_ms,
                 normalize=False):
        """Initialize the AudioConfig class.

        Args:
          sample_rate: an integer denoting the sample rate of the input waveform.
          window_ms: an integer for the length of a spectrogram frame, in ms.
          stride_ms: an integer for the frame stride, in ms.
          normalize: a boolean for whether apply normalization on the audio feature.
        """

        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.normalize = normalize


class DatasetConfig(object):
    """Config class for generating the DeepSpeechDataset."""

    def __init__(self, audio_config, data_path, characters_file_path, sortagrad):
        """Initialize the configs for deep speech dataset.

        Args:
          audio_config: AudioConfig object specifying the audio-related configs.
          metadata_path: a string denoting the full path of a manifest file.
          characters_file_path: a string specifying the vocabulary file path.
          sortagrad: a boolean, if set to true, audio sequences will be fed by
                    increasing length in the first training epoch, which will
                    expedite network convergence.

        Raises:
          RuntimeError: file path not exist.
        """

        self.audio_config = audio_config
        self.data_path = data_path
        self.characters_file_path = characters_file_path
        self.sortagrad = sortagrad