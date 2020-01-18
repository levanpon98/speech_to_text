class AudioFeaturizer(object):
    """Class to extract spectrogram features from the audio input."""

    def __init__(self,
                 sample_rate=16000,
                 window_ms=20.0,
                 stride_ms=10.0):
        """Initialize the audio featurizer class according to the configs.

        :param sample_rate: an integer specifying the sample rate of the input waveform.
        :param window_ms: an integer for the length of a spectrogram frame, in ms.
        :param stride_ms: an integer for the frame stride, in ms.
        :param normalize: a boolean for whether apply normalization on the audio feature.
        """
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
