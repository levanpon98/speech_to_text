import argparse
import os
from absl import logging
import tensorflow as tf
from utils.config import AudioConfig, DatasetConfig
from utils.dataset import DeepSpeechDataset, input_fn
from utils import fn
from model import *
import decoder

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_META_CSV = os.path.join(_BASE_DIR, 'meta.csv')
_DEFAULT_CHARACTERS_FILE = os.path.join(_BASE_DIR, 'characters.txt')
_DEFAULT_MODEL_DIR = os.path.join(_BASE_DIR, 'discriminative_model')
_DEFAULT_TRAIN_DIR = "D:/Kaggle/data/speech_to_text_vietnamese/train/"
_DEFAULT_EVAL_DIR = "D:/Kaggle/data/speech_to_text_vietnamese/test/"
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
                                                   sequence_length=ctc_input_length), axis=1)


def generate_dataset(data_dir, sample_rate,
                     window_ms, stride_ms, characters_file, sortagrad):
    audio_conf = AudioConfig(sample_rate=sample_rate,
                             window_ms=window_ms,
                             stride_ms=stride_ms,
                             normalize=True)
    data_conf = DatasetConfig(
        audio_config=audio_conf,
        data_path=data_dir,
        characters_file_path=characters_file,
        sortagrad=sortagrad
    )

    speech_dataset = DeepSpeechDataset(data_conf)
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
    targets = [entry[2] for entry in entries]  # The ground truth transcript

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
    rnn_hidden_layers = params["rnn_hidden_layers"]
    rnn_type = params["rnn_type"]
    rnn_hidden_size = params["rnn_hidden_size"]
    is_bidirectional = params["is_bidirectional"]
    use_bias = params["use_bias"]
    learning_rate = params["learning_rate"]

    input_length = features["input_length"]
    label_length = features["label_length"]
    features = features["features"]

    # Create DeepSpeech2 model.
    model = DeepSpeech2(
        rnn_hidden_layers, rnn_type,
        is_bidirectional, rnn_hidden_size,
        num_classes, use_bias)

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

    print('In training mode.')
    logits = model(features, training=True)
    probs = tf.nn.softmax(logits)
    ctc_input_length = fn.compute_length_after_conv(
        tf.shape(features)[1], tf.shape(probs)[1], input_length)
    print('# Compute CTC loss')
    loss = tf.reduce_mean(ctc_loss(label_length, ctc_input_length, labels, probs))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # global_step = tf.compat.v1.train.get_or_create_global_step()
    # minimize_op = optimizer.minimize(loss, global_step=global_step)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # # Create the train_op that groups both minimize_ops and update_ops
    # train_op = tf.group(minimize_op, update_ops)

    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    # Get both the unconditional updates (the None part)
    # and the input-conditional updates (the features part).
    update_ops = model.get_updates_for(None) + model.get_updates_for(features)
    # Compute the minimize_op.
    minimize_op = optimizer.get_updates(
        loss,
        model.trainable_variables)[0]
    train_op = tf.group(minimize_op, *update_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dir', type=str, dest='train_dir',
                        default=_DEFAULT_TRAIN_DIR,
                        help='')
    parser.add_argument('--eval-dir', type=str, dest='eval_dir',
                        default=_DEFAULT_EVAL_DIR,
                        help='')
    parser.add_argument('-sr', '--sample-rate', type=int, dest='sample_rate',
                        default=16000,
                        help='The sample rate for audio')
    parser.add_argument('--window-ms', type=int, default=20, dest='window_ms',
                        help='The frame length for spectrogram')
    parser.add_argument('--stride-ms', type=int, default=10, dest='stride_ms',
                        help='The frame step')
    parser.add_argument('--characters-file', type=str, dest='characters_file',
                        default=_DEFAULT_CHARACTERS_FILE,
                        help='The file path of characters file')
    parser.add_argument('--model-dir', type=str, dest='model_dir',
                        default=_DEFAULT_MODEL_DIR,
                        help='The file path of characters file')
    parser.add_argument('--rnn-hidden-size', type=int, default=800, dest='rnn_hidden_size',
                        help='The hidden size of RNNs.')
    parser.add_argument('--rnn-hidden-layers', type=int, default=5, dest='rnn_hidden_layers',
                        help='The number of RNN layers.')
    parser.add_argument('--use-bias', type=bool, default=True, dest='use_bias',
                        help='Use bias in the last fully-connected layer')
    parser.add_argument('--is-bidirectional', type=bool, default=True, dest='is_bidirectional',
                        help='If rnn unit is bidirectional')
    parser.add_argument('--rnn-type', type=str, choices=('lstm', 'gru'), default='gru', dest='rnn_type',
                        help='Type of RNN cell.')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=5e-4, dest='learning_rate',
                        help='The initial learning rate.')
    parser.add_argument('--batch-size', type=int,
                        default=16, dest='batch_size',
                        help='The initial batch size.')
    parser.add_argument('--train-epochs', type=int,
                        default=100, dest='train_epochs',
                        help='The initial train epochs')
    parser.add_argument('--epochs-between-evals', type=int,
                        default=2, dest='epochs_between_evals',
                        help='The initial train epochs')
    parser.add_argument('--sortagrad', type=bool, default=True, dest='sortagrad',
                        help='If true, sort examples by audio length and perform no '
                             'batch_wise shuffling for the first epoch.')
    parser.add_argument('--wer_threshold', default=None, dest='wer_threshold',
                        help='If passed, training will stop when the evaluation metric WER is  '
                             'greater than or equal to wer_threshold. For libri speech dataset '
                             'the desired wer_threshold is 0.23 which is the result achieved by '
                             ' MLPerf implementation.')

    return parser.parse_args()


def main(args):
    print("Data pre-processing...")
    train_dataset = generate_dataset(args.train_dir,
                                     args.sample_rate,
                                     args.window_ms,
                                     args.stride_ms,
                                     args.characters_file,
                                     args.sortagrad)
    eval_dataset = generate_dataset(args.eval_dir,
                                    args.sample_rate,
                                    args.window_ms,
                                    args.stride_ms,
                                    args.characters_file,
                                    args.sortagrad)
    num_classes = len(train_dataset.speech_labels)

    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    config = tf.estimator.RunConfig(train_distribute=strategy)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        config=config,
        params={
            "num_classes": num_classes,
            "rnn_hidden_layers": args.rnn_hidden_layers,
            "rnn_type": args.rnn_type,
            "rnn_hidden_size": args.rnn_hidden_size,
            "is_bidirectional": args.is_bidirectional,
            "use_bias": args.use_bias,
            "learning_rate": args.learning_rate,
        }
    )

    per_replica_batch_size = fn.per_replica_batch_size(args.batch_size, 1)

    def input_fn_train():
        return input_fn(per_replica_batch_size, train_dataset)

    def input_fn_eval():
        return input_fn(per_replica_batch_size, eval_dataset)

    total_training_cycle = (args.train_epochs // args.epochs_between_evals)

    for cycle_index in range(total_training_cycle):
        print('starting a training cycle: %d/%d', cycle_index + 1, total_training_cycle)

        train_dataset.entries = fn.batch_wise_dataset_shuffle(
            train_dataset.entries, cycle_index, args.sortagrad,
            args.batch_size)

        estimator.train(input_fn=input_fn_train)

        eval_results = evaluate_model(
            estimator, eval_dataset.speech_labels,
            eval_dataset.entries, input_fn_eval)

        print("Iteration {}: WER = {:.2f}, CER = {:.2f}".format(
            cycle_index + 1, eval_results[_WER_KEY], eval_results[_CER_KEY]))

        if fn.past_stop_threshold(
                args.wer_threshold, eval_results[_WER_KEY]):
            break


if __name__ == '__main__':
    main(parse_args())
