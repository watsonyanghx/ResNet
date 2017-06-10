"""ResNet Train/Eval/Infer module.
"""

import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf

import helper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'train ,eval or infer.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('train_labels_path', '',
                           'Filepattern for training labels.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('eval_labels_path', '',
                           'Filepattern for eval labels')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval. \
                            Change this according to your own validation data size.')
tf.app.flags.DEFINE_string('infer_data_path', '',
                           'Filepattern for infer data')
tf.app.flags.DEFINE_integer('infer_batch_count', 3000,
                            'Number of batches to eval. \
                            Change this according to your own infering data size.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')


def train(hps, X_train, y_train):
    """Training loop.

    Args:
        hps: Hyperparameters.
        X_train: Pathes of images. A 1-D numpy array of shape (N_train, ).
        y_train: Labels. A 1-D numpy array of shape (N_train, ).
    """
    images, labels = cifar_input.build_input(X_train, y_train, hps, FLAGS.mode)

    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()

    # Must right after model is established
    exclude = ['logit']
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude)

    # param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.
    #         TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    # sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge([model.summaries,
                                     tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.01

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 40000:
                self._lrn_rate = 0.1
            elif train_step < 60000:
                self._lrn_rate = 0.01
            elif train_step < 100000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.001

    saver = tf.train.Saver(variables_to_restore)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.log_root,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            save_checkpoint_secs=60,
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:

        # saver.restore(mon_sess, '/home/yang/Downloads/FILE/ml/ic/tmp/resnet_model/model.ckpt-40960')

        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)


def evaluate(hps, X_val, y_val):
    """Eval loop.
    
    Args:
        hps: Hyperparameters.
        X_val: Pathes of images. A 1-D numpy array of shape (N_val, ).
        y_val: Labels. A 1-D numpy array of shape (N_val, ).
    """
    images, labels = cifar_input.build_input(X_val, y_val, hps, FLAGS.mode)

    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step])

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(10)


def infer(hps, X_infer, y_infer):
    """Infering process

    Args:
        hps: Hyperparameters.
        X_infer: Pathes of images. A 1-D numpy array of shape (N_infer, ).
        y_infer: Labels. A 1-D numpy array of shape (N_infer, ).
                 Note that there is no labels when infering, so 'y_infer' is never used
                 in infering process, but working as a placeholder because of the requirement
                 that labels must be provided to build model.
    """

    images, labels = cifar_input.build_infer_input(X_infer, y_infer, hps)

    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()

    saver = tf.train.Saver()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    try:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    for _ in six.moves.range(FLAGS.infer_batch_count):
        predictions = sess.run(model.predictions)
        predictions = np.argmax(predictions, axis=1)

        # Store the prediction into a .txt file
        with open('./predict.txt', 'a') as f:
            for item in predictions.tolist():
                f.write(str(item) + '\n')


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if FLAGS.mode == 'train':
        batch_size = 32
    elif FLAGS.mode == 'eval':
        batch_size = 50
    elif FLAGS.mode == 'infer':
        batch_size = 100

    # Change values bellow based on your own setting.
    hps = resnet_model.HParams(batch_size=batch_size,
                               image_size=32,
                               depth=3,
                               num_classes=10,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.01,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.004,
                               relu_leakiness=0.1,
                               optimizer='mom',
                               fine_tune=False)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            X_train = helper.load_data(FLAGS.train_data_path)
            y_train = helper.load_data(FLAGS.train_labels_path)
            y_train = y_train - 1

            train(hps, X_train, y_train)
        elif FLAGS.mode == 'eval':
            X_val = helper.load_data(FLAGS.eval_data_path)
            y_val = helper.load_data(FLAGS.eval_labels_path)
            y_val = y_val - 1

            evaluate(hps, X_val, y_val)
        elif FLAGS.mode == 'infer':
            X_infer = helper.load_data(FLAGS.infer_data_path)
            y_infer = np.ones((X_infer.shape[0],))

            infer(hps, X_infer, y_infer)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
