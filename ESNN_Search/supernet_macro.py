import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import snn_supernet

"""A class of decoder to get model configuration."""
class SpaceDecoder(object):
  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments.

    E.g. r3_k5_s22_i64_o128_noskip:
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    i - input filters,
    o - output filters,
    noskip - whether there is a shortcut connection.

    Args:
      block_string: a string, representation of block arguments.

    Returns:
      A BlockArgs instance.
    Raises:
      ValueError: if the strides option is not correctly specified.
    """
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return snn_supernet.BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        id_skip=('noskip' not in block_string),
        strides=[int(options['s'][0]), int(options['s'][1])])

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of a block.

    Returns:
      A list of namedtuples to represent TBS blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

def snn_search_space(dataset_name):
  """Creates a SNN supermodel for search.

  Returns:
    blocks_args: a list of BlocksArgs for internal TBS blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  num_classes = 10
  blocks_args = [
      'r1_k3_s11_i64_o128_noskip',
      'r3_k5_s22_i128_o128',
      'r3_k5_s22_i128_o256',
      'r3_k5_s22_i256_o512'
  ]

  global_params = snn_supernet.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=num_classes,
      depth_divisor=8,
      min_depth=None,
      search_space='mnasnet')
  decoder = SpaceDecoder()
  return decoder.decode(blocks_args), global_params

def build_supernet(images, dataset_name, model_name, training, override_params=None, dropout_rate=None):
  """A helper function to create the NAS Supernet and return predicted logits and the number of spikes and SynOps.

  Args:
    images: input images tensor.
    dataset_name: string, the dataset name.
    model_name: string, the model name.
    training: boolean, whether the model is constructed for training.

  Returns:
    logits: the logits tensor of classes.
    num_of_spikes: the number of total spikes based on the threshold decisions.
    num_of_synops: the number of total SynOps based on the threshold decisions.
    indicators: save the search results of each searchable block.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  if model_name == 'single-path-search':
    blocks_args, global_params = snn_search_space(dataset_name)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    global_params = global_params._replace(**override_params)

  with tf.variable_scope(model_name):
    model = snn_supernet.SNNSuperNet(dataset_name, blocks_args, global_params, dropout_rate)
    mem_all = {}
    spike_all = {}
    TimeStep = 3
    image_inputs = images
    image_inputs = tf.cast(image_inputs, tf.float32)
    num_of_spikes = tf.constant(0.0)
    num_of_synops = tf.constant(0.0)
    for step in range(TimeStep):
        _logits, mem_all, spike_all, num_of_spikes, num_of_synops = model(image_inputs, step, mem_all, spike_all, num_of_spikes, num_of_synops, training=training)
        if step == 0:
            logits= _logits
        else:
            logits += _logits
    logits = logits / TimeStep

  logits = tf.identity(logits, 'logits')
  return logits, num_of_spikes, num_of_synops, model.indicators
