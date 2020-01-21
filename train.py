import argparse
import os

from absl import logging
import tensorflow as tf
from utils.config import AudioConfig, DatasetConfig
from utils.dataset import DeepSpeechDataset, input_fn
from utils import fn
import model as deep_speech
from absl import app as absl_app
from absl import flags
import decoder
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_META_CSV = os.path.join(_BASE_DIR, 'meta.csv')
_DEFAULT_CHARACTERS_FILE = os.path.join(_BASE_DIR, 'characters.txt')
_DEFAULT_MODEL_DIR = os.path.join(_BASE_DIR, 'model')
_DEFAULT_TRAIN_DIR = os.path.join(_BASE_DIR, "speech_to_text_vietnamese/train/")
_DEFAULT_EVAL_DIR = os.path.join(_BASE_DIR, "speech_to_text_vietnamese/test/")
_DEFAULT_SAVE_MODEL_DIR = os.path.join(_BASE_DIR, 'checkpoint')
# Evaluation metrics
_WER_KEY = "WER"
_CER_KEY = "CER"


def ctc_loss(label_length, ctc_input_length, labels, logits):
    label_length = tf.cast(tf.squeeze(label_length), dtype=tf.int32)
    ctc_input_length = tf.cast(tf.squeeze(ctc_input_length), dtype=tf.int32)
    sparse_labels = tf.cast(
        tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length),
        dtype=tf.int32
    )

    y_pred = tf.math.log(tf.transpose(logits, perm=[1, 0, 2]) + tf.keras.backend.epsilon())
    return tf.expand_dims(tf.compat.v1.nn.ctc_loss(labels=sparse_labels,
                                                   inputs=y_pred,
                                                   sequence_length=ctc_input_length,
                                                   preprocess_collapse_repeated=False,
                                                   ctc_merge_repeated=True,
                                                   ignore_longer_outputs_than_inputs=True,
                                                   time_major=True
                                                   ), axis=1)


def generate_dataset(data_dir):
    """Generate a speech dataset."""
    audio_conf = AudioConfig(sample_rate=flags_obj.sample_rate,
                             window_ms=flags_obj.window_ms,
                             stride_ms=flags_obj.stride_ms,
                             normalize=True)
    train_data_conf = DatasetConfig(
        audio_conf,
        data_dir,
        flags_obj.vocabulary_file,
        flags_obj.sortagrad
    )
    speech_dataset = DeepSpeechDataset(train_data_conf)
    return speech_dataset


def evaluate_model(estimator, speech_labels, entries, input_fn_eval):
    """Evaluate the model performance using WER anc CER as metrics.

    WER: Word Error Rate
    CER: Character Error Rate

    Args:
      estimator: estimator to evaluate.
      speech_labels: a string specifying all the character in the vocabulary.
      entries: a list of data entries (audio_file, file_size, transcript) for the
        given dataset.
      input_fn_eval: data input function for evaluation.

    Returns:
      Evaluation result containing 'wer' and 'cer' as two metrics.
    """
    # Get predictions
    predictions = estimator.predict(input_fn=input_fn_eval)

    # Get probabilities of each predicted class
    probs = [pred["probabilities"] for pred in predictions]

    num_of_examples = len(probs)
    targets = [entry[1] for entry in entries]  # The ground truth transcript

    total_wer, total_cer = 0, 0
    greedy_decoder = decoder.DeepSpeechDecoder(speech_labels)
    for i in range(num_of_examples):
        # Decode string.
        decoded_str = greedy_decoder.decode(probs[i])
        # Compute CER.
        total_cer += greedy_decoder.cer(decoded_str, targets[i]) / float(
            len(targets[i]))
        # Compute WER.
        total_wer += greedy_decoder.wer(decoded_str, targets[i]) / float(
            len(targets[i].split()))

    # Get mean value
    total_cer /= num_of_examples
    total_wer /= num_of_examples

    global_step = estimator.get_variable_value(tf.compat.v1.GraphKeys.GLOBAL_STEP)
    eval_results = {
        _WER_KEY: total_wer,
        _CER_KEY: total_cer,
        tf.compat.v1.GraphKeys.GLOBAL_STEP: global_step,
    }

    return eval_results


def model_fn(features, labels, mode, params):
    """Define model function for deep speech model.

    Args:
      features: a dictionary of input_data features. It includes the data
        input_length, label_length and the spectrogram features.
      labels: a list of labels for the input data.
      mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`.
      params: a dict of hyper parameters to be passed to model_fn.

    Returns:
      EstimatorSpec parameterized according to the input params and the
      current mode.
    """
    num_classes = params["num_classes"]
    input_length = features["input_length"]
    label_length = features["label_length"]
    features = features["features"]

    # Create DeepSpeech2 model.
    model = deep_speech.DeepSpeech2(
        flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
        flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
        num_classes, flags_obj.use_bias)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            "classes": tf.argmax(logits, axis=2),
            "probabilities": tf.nn.softmax(logits),
            "logits": logits
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    logits = model(features, training=True)
    probs = tf.nn.softmax(logits)
    ctc_input_length = fn.compute_length_after_conv(
        tf.shape(features)[1], tf.shape(probs)[1], input_length)
    # Compute CTC loss
    loss = tf.reduce_mean(ctc_loss(
        label_length, ctc_input_length, labels, probs))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=flags_obj.learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    minimize_op = optimizer.minimize(loss, global_step=global_step)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)


def run(_):
    tf.compat.v1.set_random_seed(flags_obj.seed)
    print("Data pre-processing...")
    train_speech_dataset = generate_dataset(flags_obj.train_data_dir)
    eval_speech_dataset = generate_dataset(flags_obj.eval_data_dir)
    num_classes = len(train_speech_dataset.speech_labels)

    num_gpus = flags_core.get_num_gpus(flags_obj)
    distribution_strategy = distribution_utils.get_distribution_strategy(num_gpus=num_gpus)
    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=flags_obj.model_dir,
        config=run_config,
        params={
            "num_classes": num_classes,
        }
    )

    run_params = {
        "batch_size": flags_obj.batch_size,
        "train_epochs": flags_obj.train_epochs,
        "rnn_hidden_size": flags_obj.rnn_hidden_size,
        "rnn_hidden_layers": flags_obj.rnn_hidden_layers,
        "rnn_type": flags_obj.rnn_type,
        "is_bidirectional": flags_obj.is_bidirectional,
        "use_bias": flags_obj.use_bias
    }

    dataset_name = "LibriSpeech"
    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info("deep_speech", dataset_name, run_params,
                                  test_id=flags_obj.benchmark_test_id)

    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)

    per_replica_batch_size = distribution_utils.per_replica_batch_size(
        flags_obj.batch_size, num_gpus)

    def input_fn_train():
        return input_fn(per_replica_batch_size, train_speech_dataset)

    def input_fn_eval():
        return input_fn(
            per_replica_batch_size, eval_speech_dataset)

    total_training_cycle = (flags_obj.train_epochs //
                            flags_obj.epochs_between_evals)

    for cycle_index in range(total_training_cycle):
        print('Starting a training cycle: %d/%d' % (cycle_index + 1, total_training_cycle))

        train_speech_dataset.entries = fn.batch_wise_dataset_shuffle(
            train_speech_dataset.entries, cycle_index, flags_obj.sortagrad,
            flags_obj.batch_size)

        estimator.train(input_fn=input_fn_train, hooks=train_hooks)

        eval_results = evaluate_model(
            estimator, eval_speech_dataset.speech_labels,
            eval_speech_dataset.entries, input_fn_eval)

        print("Iteration {}: WER = {:.2f}, CER = {:.2f}".format(
            cycle_index + 1, eval_results[_WER_KEY], eval_results[_CER_KEY]))

        if model_helpers.past_stop_threshold(
                flags_obj.wer_threshold, eval_results[_WER_KEY]):
            break


def define_deep_speech_flags():
    """Add flags for run_deep_speech."""
    # Add common flags
    flags_core.define_base(
        data_dir=False,  # we use train_data_dir and eval_data_dir instead
        export_dir=True,
        train_epochs=True,
        hooks=True,
        epochs_between_evals=True,
    )
    flags_core.define_performance(
        num_parallel_calls=False,
        inter_op=False,
        intra_op=False,
        synthetic_data=False,
        max_train_steps=False,
        dtype=False
    )
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags_core.set_defaults(
        model_dir=_DEFAULT_MODEL_DIR,
        export_dir=_DEFAULT_SAVE_MODEL_DIR,
        train_epochs=10,
        batch_size=128,
        hooks=[],
        epochs_between_evals=4)

    # Deep speech flags
    flags.DEFINE_integer(
        name="seed", default=1,
        help=flags_core.help_wrap("The random seed."))

    flags.DEFINE_string(
        name="train_data_dir",
        default=_DEFAULT_TRAIN_DIR,
        help=flags_core.help_wrap("The csv file path of train dataset."))

    flags.DEFINE_string(
        name="eval_data_dir",
        default=_DEFAULT_EVAL_DIR,
        help=flags_core.help_wrap("The csv file path of evaluation dataset."))

    flags.DEFINE_bool(
        name="sortagrad", default=True,
        help=flags_core.help_wrap(
            "If true, sort examples by audio length and perform no "
            "batch_wise shuffling for the first epoch."))

    flags.DEFINE_integer(
        name="sample_rate", default=16000,
        help=flags_core.help_wrap("The sample rate for audio."))

    flags.DEFINE_integer(
        name="window_ms", default=20,
        help=flags_core.help_wrap("The frame length for spectrogram."))

    flags.DEFINE_integer(
        name="stride_ms", default=10,
        help=flags_core.help_wrap("The frame step."))

    flags.DEFINE_string(
        name="vocabulary_file", default=_DEFAULT_CHARACTERS_FILE,
        help=flags_core.help_wrap("The file path of vocabulary file."))

    # RNN related flags
    flags.DEFINE_integer(
        name="rnn_hidden_size", default=800,
        help=flags_core.help_wrap("The hidden size of RNNs."))

    flags.DEFINE_integer(
        name="rnn_hidden_layers", default=5,
        help=flags_core.help_wrap("The number of RNN layers."))

    flags.DEFINE_bool(
        name="use_bias", default=True,
        help=flags_core.help_wrap("Use bias in the last fully-connected layer"))

    flags.DEFINE_bool(
        name="is_bidirectional", default=True,
        help=flags_core.help_wrap("If rnn unit is bidirectional"))

    flags.DEFINE_enum(
        name="rnn_type", default="gru",
        enum_values=deep_speech.SUPPORTED_RNNS.keys(),
        case_sensitive=False,
        help=flags_core.help_wrap("Type of RNN cell."))

    # Training related flags
    flags.DEFINE_float(
        name="learning_rate", default=5e-4,
        help=flags_core.help_wrap("The initial learning rate."))

    # Evaluation metrics threshold
    flags.DEFINE_float(
        name="wer_threshold", default=None,
        help=flags_core.help_wrap(
            "If passed, training will stop when the evaluation metric WER is "
            "greater than or equal to wer_threshold. For libri speech dataset "
            "the desired wer_threshold is 0.23 which is the result achieved by "
            "MLPerf implementation."))

    flags.DEFINE_integer(
        name='num_gpus', default=-1,
        help='num_gpus'
    )


def main(_):
    with logger.benchmark_context(flags_obj):
        run(flags_obj)


if __name__ == "__main__":
    define_deep_speech_flags()
    flags_obj = flags.FLAGS
    absl_app.run(main)
