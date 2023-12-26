from absl import flags
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def Params_Generate():
    FLAGS = tf.app.flags.FLAGS

    flags.DEFINE_string(
        'dataset_name', default='CIFAR10',
        help='The dataset name.')

    flags.DEFINE_integer(
        'seed', default=2023, help='Random seed setting.')

    flags.DEFINE_string(
        'data_dir', default='./cifar-10-python/',
        help='The directory where the input data is stored.')

    flags.DEFINE_string(
        'model_dir', default='./NAS_train_final_results/',
        help='The directory where the model and training/evaluation summaries are stored.')

    flags.DEFINE_string(
        'model_name',
        default='single-path',
        help='The model name.')

    flags.DEFINE_string(
        'mode', default='train_and_eval',
        help='One of {"train_and_eval", "train", "eval"}.')

    flags.DEFINE_integer(
        'train_steps', default=700000,
        help=('The number of steps to use for training. Default is 700000 steps'
              'which is 350 epochs at batch size 25 and training dataset size 50000.'
              'The number of training steps for an epoch is calculated as follows:'
              'training dataset size  / batch size, i.e. 50000 / 25 is 2000.'
              'This flag should be adjusted according to the --train_batch_size flag.'))

    flags.DEFINE_integer(
        'batch_size', default=25, help='Batch size for dataset.')

    flags.DEFINE_integer(
        'train_batch_size', default=25, help='Batch size for training.')

    flags.DEFINE_integer(
        'eval_batch_size', default=25, help='Batch size for evaluation.')

    flags.DEFINE_integer(
        'num_train_images', default=50000, help='Size of training data set.')

    flags.DEFINE_integer(
        'num_eval_images', default=10000, help='Size of evaluation data set.')

    flags.DEFINE_integer(
        'steps_per_eval', default=2000,
        help=('Controls how often evaluation is performed. Since evaluation is'
              ' fairly expensive, it is advised to evaluate as infrequently as'
              ' possible (i.e. up to --train_steps, which evaluates the model only'
              ' after finishing the entire training regime).'))

    flags.DEFINE_integer(
        'eval_timeout',
        default=None,
        help='Maximum seconds between checkpoints before evaluation terminates.')

    flags.DEFINE_integer(
        'iterations_per_loop', default=2000,
        help=('The number of steps to run before the next checkpoint saving.'))

    flags.DEFINE_string(
        'data_format', default='channels_last',
        help=('A flag to override the data format used in the model. The value'
              ' is either channels_first or channels_last.'))

    flags.DEFINE_integer(
        'num_label_classes', default=10, help='Number of classes for the dataset.')

    flags.DEFINE_float(
        'base_learning_rate',
        default=0.00015625,
        help='Base learning rate when train batch size is 25.')

    flags.DEFINE_float(
        'momentum', default=0.9,
        help=('Momentum parameter used in the MomentumOptimizer.'))

    flags.DEFINE_float(
        'moving_average_decay', default=0.9999,
        help=('Moving average decay rate.'))

    flags.DEFINE_float(
        'weight_decay', default=1e-5,
        help=('Weight decay coefficiant for l2 regularization.'))

    flags.DEFINE_float(
        'label_smoothing', default=0.1,
        help=('Label smoothing parameter used in the softmax_cross_entropy'))

    flags.DEFINE_float(
        'dropout_rate', default=0.2,
        help=('Dropout rate for the final output layer.'))

    flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                                                     'which the global step information is logged.')

    flags.DEFINE_string(
        'parse_search_dir',
        default='./NAS_search_results/',
        help=('The directory where the output of ESNN searched model is stored.'))

    return FLAGS