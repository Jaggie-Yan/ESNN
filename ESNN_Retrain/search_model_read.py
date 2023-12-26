import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model_def

"""A class of decoder to get the searched model configuration."""
class NetDecoder(object):
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

    return model_def.BlockArgs(
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

"""Loading model data using tensorboard"""
import parse_netarch
def parse_netarch_model(parse_lambda_dir):
  # tensorboard sampling settings
  tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
  }
  indicator_values, len_num = parse_netarch.parse_indicators_single_path_nas(parse_lambda_dir, tf_size_guidance)
  network, block_inds_last, thresh_last = parse_netarch.encode_single_path_nas_arch(indicator_values, len_num)
  parse_netarch.print_net(network)
  blocks_args = parse_netarch.convnet_encoder(network)
  decoder = NetDecoder()
  return decoder.decode(blocks_args[-1]), block_inds_last, thresh_last