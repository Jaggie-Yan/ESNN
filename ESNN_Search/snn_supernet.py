import collections
from six.moves import xrange
from superkernel import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_divisor', 'min_depth', 'search_space',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'id_skip', 'strides'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

"""Gradient calculation for SNN."""
@tf.custom_gradient
def SpikeAct(input):
    cond_org = input <= 0
    cond_one = input > 0
    ones = tf.ones_like(input)
    zeros = tf.zeros_like(input)
    y = tf.where(cond_org, input, ones)
    y = tf.where(cond_one, y, zeros)

    def grad(dy):
        cond_one = abs(input) < 0.5
        zeros = tf.zeros_like(dy)
        return tf.where(cond_one, dy, zeros)
    return y, grad

"""Approximation firing function."""
def SpikeFunction(input, mem, spike, thresh, decay=0.2):
    mem = mem * decay * (1. - spike) + input
    spike = SpikeAct(mem - thresh)
    return mem, spike

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable.
    dtype: dtype of variable.
    partition_info: unused.

  Returns:
    An initialization for the variable.
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable.
    dtype: dtype of variable.
    partition_info: unused.

  Returns:
    An initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)

"""Searchable layer model based on the spiking residual block (SRB)."""
class TBS(tf.keras.Model):
  def __init__(self, block_args, global_params, dropout_rate, thresh, count):
    """Initialize a TBS block.

    Args:
      block_args: BlockArgs, arguments to create a TBS Block.
      global_params: GlobalParams, a set of global parameters.
    """
    super(TBS, self).__init__()
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]

    self.dropout_rate = dropout_rate
    self.thresh = thresh
    self.count = count
    self._search_space = global_params.search_space
    self._build()

  # build the block according to arguments
  def _build(self):
    # the first conv in SRB
    self.head_conv_kernel = self.add_weight(shape=(self._block_args.kernel_size, self._block_args.kernel_size, \
                                            self._block_args.input_filters, self._block_args.output_filters),
                                            initializer=tf.random_normal_initializer(
                                                mean=0.0,
                                                stddev=np.sqrt(2.0 / int(
                                                    self._block_args.kernel_size * self._block_args.kernel_size * self._block_args.output_filters)),
                                                dtype=tf.float32),
                                            name="head_conv_kernel")
    # the second conv in SRB
    self.tail_conv_kernel = self.add_weight(shape=(self._block_args.kernel_size, self._block_args.kernel_size,\
                                            self._block_args.output_filters, self._block_args.output_filters),
                                             initializer=tf.random_normal_initializer(
                                                 mean=0.0,
                                                 stddev=np.sqrt(2.0 / int(
                                                     self._block_args.kernel_size * self._block_args.kernel_size * self._block_args.output_filters)),
                                                 dtype=tf.float32),
                                             name="tial_conv_kernel")

    # construct branchless searchable superkernels
    self.kernel_masked_head = KernelMasked(kernel=self.head_conv_kernel, strides=self._block_args.strides, dropout_rate=self.dropout_rate)
    self.kernel_masked_tail = KernelMasked(kernel=self.tail_conv_kernel, strides=[1, 1], dropout_rate=self.dropout_rate)

    self._bn0 = tf.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)
    self._bn1 = tf.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)

  def call(self, inputs, timestep, mem_all, spike_all, num_of_spikes, training=True):
    """Implementation of TBS call().

    Args:
      inputs: the inputs tensor.
      timestep: the current timestep of our searched SNN.
      mem_all: save the voltage information at the previous moment on each layer.
      spike_all: save the spiking information at the previous moment on each layer.
      num_of_spikes: the current number of spikes of our searched SNN.
      training: boolean, whether the model is constructed for training.

    Returns:
      Output tensors and structure imformation of the current block.
    """
    head_conv_masked = self.kernel_masked_head.call()
    tail_conv_masked = self.kernel_masked_tail.call()
    head_skip_flag = tf.cond(tf.abs(tf.reduce_sum(head_conv_masked)) > 0, lambda: 0, lambda: 1)
    tail_skip_flag = tf.cond(tf.abs(tf.reduce_sum(tail_conv_masked)) > 0, lambda: 0, lambda: 1)

    x = tf.nn.conv2d(inputs,
                     head_conv_masked,
                     strides=self._block_args.strides,
                     padding='SAME')
    x = self._bn0(x, training=training)
    if timestep == 0:
        _head_conv_mem = x
        _head_conv_spike = SpikeAct(x - self.thresh)
        mem_all['_head_conv_mem' + str(self.count)] = _head_conv_mem
        spike_all['_head_conv_spike' + str(self.count)] = _head_conv_spike
    else:
        _head_conv_mem, _head_conv_spike = SpikeFunction(x, mem_all['_head_conv_mem' + str(self.count)],
                                                         spike_all['_head_conv_spike' + str(self.count)], self.thresh)
        mem_all['_head_conv_mem' + str(self.count)] = _head_conv_mem
        spike_all['_head_conv_spike' + str(self.count)] = _head_conv_spike
    x = spike_all['_head_conv_spike' + str(self.count)]
    x_head = x
    num_of_spikes += tf.reduce_sum(x)
    tf.logging.info('head conv outputs: %s shape: %s' % (x.name, x.shape))

    # determine whether the first conv in SRB is skipped
    if self._block_args.id_skip and all(s == 1 for s in self._block_args.strides):
        x = tf.cond(head_skip_flag > 0, lambda: inputs, lambda: x)
        x_head = x

    x = tf.nn.conv2d(x_head,
                     tail_conv_masked,
                     strides=[1, 1],
                     padding='SAME')
    x = self._bn1(x, training=training)
    if timestep == 0:
        _tail_conv_mem = x
        _tail_conv_spike = SpikeAct(x - self.thresh)
        mem_all['_tail_conv_mem' + str(self.count)] = _tail_conv_mem
        spike_all['_tail_conv_spike' + str(self.count)] = _tail_conv_spike
    else:
        _tail_conv_mem, _tail_conv_spike = SpikeFunction(x, mem_all['_tail_conv_mem' + str(self.count)],
                                                         spike_all['_tail_conv_spike' + str(self.count)],
                                                         self.thresh)
        mem_all['_tail_conv_mem' + str(self.count)] = _tail_conv_mem
        spike_all['_tail_conv_spike' + str(self.count)] = _tail_conv_spike
    x = spike_all['_tail_conv_spike' + str(self.count)]
    num_of_spikes += tf.reduce_sum(x)
    tf.logging.info('tail conv outputs: %s shape: %s' % (x.name, x.shape))

    # determine whether the second conv in SRB is skipped
    x = tf.cond(head_skip_flag > 0, lambda: tf.cond(tail_skip_flag > 0, lambda: (x * 0.0), lambda: x),
                lambda: tf.cond(tail_skip_flag > 0, lambda: x_head, lambda: x))

    # shortcut connection in SRB
    if self._block_args.id_skip:
        if all(s == 1 for s in self._block_args.strides):
            x += inputs
    return x, x_head, num_of_spikes, self._block_args.kernel_size, self._block_args.strides[0], self._block_args.output_filters

"""Class implements tf.keras.Model for our searched SNN with superkernels."""
class SNNSuperNet(tf.keras.Model):
    def __init__(self, dataset_name, blocks_args=None, global_params=None,dropout_rate=None):
        """Initializes a searched SNN instance.

        Args:
          blocks_args: A list of BlockArgs to construct TBS block modules.
          global_params: GlobalParams, a set of global parameters.

        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(SNNSuperNet, self).__init__()
        if not isinstance(blocks_args, list):
          raise ValueError('blocks_args should be a list.')
        self.dataset_name = dataset_name
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.dropout_rate = dropout_rate
        self._search_space = global_params.search_space

        assert self._search_space == 'mnasnet'
        # threshold is trainable in our method
        self.thresh = self.add_weight(shape=(1,),
                                      initializer=tf.random_normal_initializer(
                                          mean=0.5,
                                          stddev=0),
                                      name="thresh")
        self._build()

    # build the searchable supernet
    def _build(self):
        self._blocks = []
        count = -1
        # Builds blocks.
        for block_args in self._blocks_args:
          assert block_args.num_repeat > 0
          count += 1
          self._blocks.append(TBS(block_args, self._global_params, self.dropout_rate, self.thresh, count))
          # if not the first block update block input and output filters
          if block_args.num_repeat > 1:
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
          for _ in xrange(block_args.num_repeat - 1):
            count += 1
            self._blocks.append(TBS(block_args, self._global_params, self.dropout_rate, self.thresh, count))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
          channel_axis = 1
        else:
          channel_axis = -1

        # set channels for stem blocks and tail blocks according to the dataset name
        self.stem_channel = 64
        self.tail_channel = 1024

        # stem blocks
        self._conv_stem = tf.keras.layers.Conv2D(
            filters=self.stem_channel,
            kernel_size=[3, 3],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)

        # tail blocks
        self._conv_head = tf.keras.layers.Conv2D(
            filters=self.tail_channel,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self._global_params.data_format)

        self._fc = tf.keras.layers.Dense(
            self._global_params.num_classes,
            kernel_initializer=dense_kernel_initializer)

    def call(self, inputs, timestep, mem_all, spike_all, num_of_spikes, num_of_synops, training=True):
        """Implementation of SNNSuperNet call().

        Args:
          inputs: the inputs tensor.
          timestep: the current timestep of our searched SNN.
          mem_all: save the voltage information at the previous moment on each layer.
          spike_all: save the spiking information at the previous moment on each layer.
          num_of_spikes: the current number of spikes of our searched SNN.
          num_of_synops: the current number of SynOps of our searched SNN.
          training: boolean, whether the model is constructed for training.

        Returns:
          output tensors, the voltage and spiking information of each layer and the number of total spikes and SynOps based on the threshold decisions.
        """
        self.indicators = {}
        # call stem blocks
        with tf.variable_scope('mnas_stem'):
            outputs = self._conv_stem(inputs)
            outputs = self._bn0(outputs, training=training)
            if timestep == 0:
                _conv_stem_mem = outputs
                _conv_stem_spike = SpikeAct(outputs - self.thresh)
                mem_all['_conv_stem_mem'] = _conv_stem_mem
                spike_all['_conv_stem_spike'] = _conv_stem_spike
            else:
                _conv_stem_mem, _conv_stem_spike = SpikeFunction(outputs, mem_all['_conv_stem_mem'],
                                                                 spike_all['_conv_stem_spike'], self.thresh)
                mem_all['_conv_stem_mem'] = _conv_stem_mem
                spike_all['_conv_stem_spike'] = _conv_stem_spike

            outputs = spike_all['_conv_stem_spike']
            num_of_spikes += tf.reduce_sum(outputs)
        tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)

        # call TBS blocks
        for idx, block in enumerate(self._blocks):
            with tf.variable_scope('mnas_blocks_%s' % idx):
                layer_inputs = outputs
                outputs, outputs_head, num_of_spikes, kernel_size, stride, out_channels = block.call(outputs, timestep, mem_all, spike_all,
                                                    num_of_spikes, training=training)
                if kernel_size == 5:
                    searched_kernel_size_head = block.kernel_masked_head.d3x3 * (3 + block.kernel_masked_head.d5x5 * 2)
                    num_of_synops += tf.reduce_sum(layer_inputs * out_channels * tf.square(searched_kernel_size_head / stride))
                    searched_kernel_size_tail = block.kernel_masked_tail.d3x3 * (3 + block.kernel_masked_tail.d5x5 * 2)
                    num_of_synops += tf.reduce_sum(outputs_head * out_channels * tf.square(searched_kernel_size_tail / 1))
                    self.indicators['block_%s' % idx] = {
                        'd3x3_head': block.kernel_masked_head.d3x3,
                        'd5x5_head': block.kernel_masked_head.d5x5,
                        'd3x3_tail': block.kernel_masked_tail.d3x3,
                        'd5x5_tail': block.kernel_masked_tail.d5x5,
                        'dropout_rate': block.kernel_masked_head.dropout_rate,
                        't3x3_head': block.kernel_masked_head.t3x3,
                        't5x5_head': block.kernel_masked_head.t5x5,
                        't3x3_tail': block.kernel_masked_tail.t3x3,
                        't5x5_tail': block.kernel_masked_tail.t5x5,
                        'thresh': self.thresh,
                        'num_of_spikes': num_of_spikes,
                        'num_of_synops': num_of_synops
                    }
                else:
                    num_of_synops += tf.reduce_sum(layer_inputs * out_channels * tf.square(kernel_size / stride))
                    num_of_synops += tf.reduce_sum(outputs_head * out_channels * tf.square(kernel_size / 1))

        # call tail blocks
        with tf.variable_scope('mnas_head'):
            num_of_synops += tf.reduce_sum(outputs * self.tail_channel * tf.square(1 / 1))
            outputs = self._conv_head(outputs)
            outputs = self._bn1(outputs, training=training)
            if timestep == 0:
                _conv_head_mem = outputs
                _conv_head_spike = SpikeAct(outputs - self.thresh)
                mem_all['_conv_head_mem'] = _conv_head_mem
                spike_all['_conv_head_spike'] = _conv_head_spike
            else:
                _conv_head_mem, _conv_head_spike = SpikeFunction(outputs,
                                                                 mem_all['_conv_head_mem'],
                                                                 spike_all['_conv_head_spike'], self.thresh)
                mem_all['_conv_head_mem'] = _conv_head_mem
                spike_all['_conv_head_spike'] = _conv_head_spike
            outputs = spike_all['_conv_head_spike']
            num_of_spikes += tf.reduce_sum(outputs)

            num_of_synops += tf.reduce_sum(outputs)
            outputs = self._avg_pooling(outputs)
            if timestep == 0:
                _avg_pooling_mem = outputs
                _avg_pooling_spike = SpikeAct(outputs - self.thresh)
                mem_all['_avg_pooling_mem'] = _avg_pooling_mem
                spike_all['_avg_pooling_spike'] = _avg_pooling_spike
            else:
                _avg_pooling_mem, _avg_pooling_spike = SpikeFunction(outputs,
                                                             mem_all['_avg_pooling_mem'],
                                                             spike_all['_avg_pooling_spike'], self.thresh)
                mem_all['_avg_pooling_mem'] = _avg_pooling_mem
                spike_all['_avg_pooling_spike'] = _avg_pooling_spike
            outputs = spike_all['_avg_pooling_spike']
            num_of_spikes += tf.reduce_sum(outputs)

            num_of_synops += tf.reduce_sum(outputs * self._global_params.num_classes)
            outputs = self._fc(outputs)
            if timestep == 0:
                _fc_mem = outputs
                _fc_spike = SpikeAct(outputs - self.thresh)
                mem_all['_fc_mem'] = _fc_mem
                spike_all['_fc_spike'] = _fc_spike
            else:
                _fc_mem, _fc_spike = SpikeFunction(outputs, mem_all['_fc_mem'],
                                                   spike_all['_fc_spike'], self.thresh)
                mem_all['_fc_mem'] = _fc_mem
                spike_all['_fc_spike'] = _fc_spike

            num_of_spikes += tf.reduce_sum(spike_all['_fc_spike'])
            outputs = mem_all['_fc_mem']
        return outputs, mem_all, spike_all, num_of_spikes, num_of_synops