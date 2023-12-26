import os
import time
from absl import app
import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from cifar10_dataset_input import CIFAR10_Dataset_Input
import supernet_macro
import nas_utils
import params_generate
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
from tensorflow.python.keras import backend as K

# parameter loading
FLAGS = params_generate.Params_Generate()

# normalization parameters for different datasets
MEAN_RGB = [0.493 * 255, 0.484 * 255, 0.448 * 255]
STDDEV_RGB = [0.247 * 255, 0.243 * 255, 0.262 * 255]

def nas_model_fn(features, labels, mode, params):
  """The model_fn for NAS search.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples.
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`.
    params: `dict` of parameters passed to the model from the Estimator.

  Returns:
    A `EstimatorSpec` for the model.
  """
  if isinstance(features, dict):
    input_images = features['feature']
  else:
    input_images = features
  input_images = tf.cast(input_images, dtype=tf.float32)

  # normalize the image to zero mean and unit variance
  input_images -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=input_images.dtype)
  input_images /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=input_images.dtype)

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  has_moving_average_decay = (FLAGS.moving_average_decay > 0)

  # this is essential if using a keras-derived model
  K.set_learning_phase(is_training)
  tf.logging.info('Using open-source implementation for NAS definition.')

  global_step = tf.train.get_global_step()
  warmup_steps = FLAGS.warmup_steps
  dropout_rate = nas_utils.build_dropout_rate(global_step, warmup_steps)

  # get predicted logits of the current searched architecture
  logits, num_of_spikes, num_of_synops, indicators = supernet_macro.build_supernet(
      input_images,
      dataset_name=FLAGS.dataset_name,
      model_name=FLAGS.model_name,
      training=is_training,
      override_params=None,
      dropout_rate=dropout_rate)
  num_of_spikes /= FLAGS.batch_size
  num_of_synops /= FLAGS.batch_size

  # predictive mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # calculate loss including softmax cross entropy and L2 regularization
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing)

  # calculate the computational cost and reshape it to match the size of cross entropy
  loss_term_lambda = nas_utils.build_loss_term_lambda(global_step,
                        warmup_steps, FLAGS.loss_term_lambda_val)
  num_of_synops_loss = loss_term_lambda * FLAGS.loss_lambda * (num_of_synops/FLAGS.beta)
  num_of_synops_loss = tf.reshape(num_of_synops_loss, shape=cross_entropy.shape)

  # add weight decay to the loss for non-batch-normalization variables
  loss = (cross_entropy + FLAGS.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])) + num_of_synops_loss

  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=FLAGS.moving_average_decay, num_updates=global_step)
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
      if 'moving_mean' in v.name or 'moving_variance' in v.name:
        ema_vars.append(v)
    ema_vars = list(set(ema_vars))

  # training mode: build learning rate and optimizer for current epoch and execute training operation
  restore_vars_dict = None
  if is_training:
    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 25.0)
    learning_rate = nas_utils.build_learning_rate(scaled_lr, global_step, params['steps_per_epoch'], warmup_epochs=-1)
    optimizer = nas_utils.build_optimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    # create scalar summaries for training metrics
    def summary_fn(loss, ce, num_of_spikes, num_of_synops, lr, T_List, labels_num):
      t_list = []
      for idx in range(int(len(T_List)/labels_num)):
          t_list.append(T_List[(idx*labels_num):(idx*labels_num+labels_num)])

      tf.summary.scalar('loss', loss[0])
      tf.summary.scalar('ce', ce[0])
      tf.summary.scalar('num_of_spikes', num_of_spikes[0])
      tf.summary.scalar('num_of_synops', num_of_synops[0])
      tf.summary.scalar('learning_rate', lr[0])
      for idx, t_ in enumerate(t_list):
        for label_, t_tensor in zip(['d3x3_head_','d5x5_head_','d3x3_tail_','d5x5_tail_','dropout_rate_',\
                                     't3x3_head_','t5x5_head_','t3x3_tail_','t5x5_tail_','thresh_',\
                                     'num_of_spikes_','num_of_synops_'], t_):
          sum_label_ = label_ + str(idx+1)
          tf.summary.scalar(sum_label_, t_tensor[0])

    loss_t = tf.reshape(loss, [1])
    ce_t = tf.reshape(cross_entropy, [1])
    num_of_spikes_t = tf.reshape(num_of_spikes, [1])
    num_of_synops_t = tf.reshape(num_of_synops, [1])
    lr_t = tf.reshape(learning_rate, [1])

    decision_labels = ['d3x3_head','d5x5_head','d3x3_tail','d5x5_tail','dropout_rate',\
                                     't3x3_head','t5x5_head','t3x3_tail','t5x5_tail','thresh',\
                                     'num_of_spikes','num_of_synops']
    t_list = []
    for idx in range(FLAGS.repeat_num):
        key_ = 'block_' + str(idx+1)
        for decision_label in decision_labels:
          v = indicators[key_][decision_label]
          t_list.append(tf.reshape(v, [1]))

    summary_fn(loss_t, ce_t, num_of_spikes_t, num_of_synops_t, lr_t, t_list, len(decision_labels))

  else:
    train_op = None
    if has_moving_average_decay:
      # load moving average variables for eval
      restore_vars_dict = ema.variables_to_restore(ema_vars)

  # evaluation mode
  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits, num_of_spikes, num_of_synops):
      """Evaluation metric function. Evaluates accuracy.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
          'num_of_spikes': (num_of_spikes, num_of_spikes),
          'num_of_synops': (num_of_synops, num_of_synops),
      }
    eval_metrics = metric_fn(labels, logits, num_of_spikes, num_of_synops)

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('number of trainable parameters: {}'.format(num_params))

  def _scaffold_fn():
    saver = tf.train.Saver(restore_vars_dict)
    return tf.train.Scaffold(saver=saver)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics,
      scaffold=_scaffold_fn() if has_moving_average_decay else None)

def main(unused_argv):
  # set randomness
  random.seed(FLAGS.seed)
  os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)
  config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True,
          gpu_options={"allow_growth": True},
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))))

  # initialize model parameters
  params = dict(steps_per_epoch=FLAGS.num_train_images / FLAGS.train_batch_size,
                batch_size=FLAGS.train_batch_size)

  nas_est = tf.estimator.Estimator(
      model_fn=nas_model_fn,
      model_dir=config.model_dir,
      config=config,
      params=params)

  # input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation
  tf.logging.info('Using dataset: %s', FLAGS.data_dir)
  image_train = CIFAR10_Dataset_Input(FLAGS, 'train')
  image_eval = CIFAR10_Dataset_Input(FLAGS, 'eval')

  if FLAGS.mode == 'eval':
    eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
    # run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()
        eval_results = nas_est.evaluate(
            input_fn=image_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                        eval_results, elapsed_time)

        # terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break

      except tf.errors.NotFoundError:
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

  else:
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)

    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', FLAGS.train_steps,
        FLAGS.train_steps / params['steps_per_epoch'], current_step)

    start_timestamp = time.time()

    if FLAGS.mode == 'train':
      hooks = []
      nas_est.train(
          input_fn=image_train.input_fn,
          max_steps=FLAGS.train_steps,
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < FLAGS.train_steps:
        # train for up to steps_per_eval number of steps
        # a checkpoint will be written to --model_dir at the end of training
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        nas_est.train(
            input_fn=image_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # evaluate the model on the most recent model in --model_dir
        tf.logging.info('Starting to evaluate.')
        eval_results = nas_est.evaluate(
            input_fn=image_eval.input_fn,
            steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      FLAGS.train_steps, elapsed_time)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)

