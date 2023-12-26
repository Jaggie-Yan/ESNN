import collections
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_divisor', 'min_depth',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'id_skip', 'strides'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

"""Read the parameters of the non-skip layers in the searched architecture found by NAS
as the initialized parameters for the retraining phase"""
from cifar10_kernel_initializer import *
kernel_initializer = {}
shortcut_initializer = {}
gamma_initializer = {}
beta_initializer = {}
kernel_initializer['0'] = kernel_initializer_all['0']
kernel_initializer['1'] = kernel_initializer_all['1']
kernel_initializer['11'] = kernel_initializer_all['11']
gamma_initializer['0'] = gamma_initializer_all['0']
gamma_initializer['1'] = gamma_initializer_all['1']
gamma_initializer['11'] = gamma_initializer_all['11']
beta_initializer['0'] = beta_initializer_all['0']
beta_initializer['1'] = beta_initializer_all['1']
beta_initializer['11'] = beta_initializer_all['11']
cnt = 1
for i in range(len(block_inds_last)):
    if block_inds_last[i][0] > 0 or block_inds_last[i][2] > 0:
        cnt += 1
        kernel_initializer[str(cnt)] = kernel_initializer_all[str(i + 2)]
        gamma_initializer[str(cnt)] = gamma_initializer_all[str(i + 2)]
        beta_initializer[str(cnt)] = beta_initializer_all[str(i + 2)]
        kernel_initializer[str(cnt) + str(1)] = kernel_initializer_all[str(i + 2) + str(1)]
        gamma_initializer[str(cnt) + str(1)] = gamma_initializer_all[str(i + 2) + str(1)]
        beta_initializer[str(cnt) + str(1)] = beta_initializer_all[str(i + 2) + str(1)]

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

"""Searched layer model found by NAS based on the spiking residual block (SRB)."""
class TBS(object):
  def __init__(self, block_args0, block_args1, global_params, thresh, count):
    """Initialize a searched TBS block.

    Args:
      block_args: BlockArgs, arguments to create a searched TBS Block.
      global_params: GlobalParams, a set of global parameters.
    """
    self._block_args0 = block_args0
    self._block_args1 = block_args1
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]

    self.thresh = thresh
    self.count = count
    self._build()

  # build the searched block according to arguments
  def _build(self):
    # the first conv in SRB
    if self._block_args0.kernel_size > 0:
        filters = self._block_args0.output_filters
        kernel_size = self._block_args0.kernel_size
        self.head_conv = tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=[kernel_size, kernel_size],
          strides=self._block_args0.strides,
          kernel_initializer=kernel_initializer[str(self.count + 1)],
          padding='same',
          use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            beta_initializer=beta_initializer[str(self.count + 1)],
            gamma_initializer=gamma_initializer[str(self.count + 1)],
            fused=True)
    # the second conv in SRB
    if self._block_args1.kernel_size > 0:
        filters = self._block_args1.output_filters
        kernel_size = self._block_args1.kernel_size
        self.tail_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=[1, 1],
            kernel_initializer=kernel_initializer[str(self.count + 1) + str(1)],
            padding='same',
            use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            beta_initializer=beta_initializer[str(self.count + 1) + str(1)],
            gamma_initializer=gamma_initializer[str(self.count + 1) + str(1)],
            fused=True)

  def call(self, inputs, timestep, mem_all, spike_all, num_of_spikes, num_of_synops, training=True):
    """Implementation of TBS call().

    Args:
      inputs: the inputs tensor.
      timestep: the current timestep of our searched SNN.
      mem_all: save the voltage information at the previous moment on each layer.
      spike_all: save the spiking information at the previous moment on each layer.
      num_of_spikes: the current number of spikes of our searched SNN.
      num_of_synops: the current number of SynOps of our searched SNN.
      training: boolean, whether the model is constructed for training.

    Returns:
      Output tensors and the current number of spikes and SynOps of our searched SNN.
    """
    if self._block_args0.kernel_size > 0:
        num_of_synops += tf.reduce_sum(inputs * self._block_args0.output_filters * tf.square(self._block_args0.kernel_size/self._block_args0.strides[0]))
        x = self.head_conv(inputs)
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
        num_of_spikes += tf.reduce_sum(x)
        tf.logging.info('head conv outputs: %s shape: %s' % (x.name, x.shape))
    else:
        x = inputs

    if self._block_args1.kernel_size > 0:
        num_of_synops += tf.reduce_sum(x * self._block_args1.output_filters * tf.square(self._block_args1.kernel_size/1))
        x = self.tail_conv(x)
        x = self._bn1(x, training=training)
        if timestep == 0:
            _tail_conv_mem = x
            _tail_conv_spike = SpikeAct(x - self.thresh)
            mem_all['_tail_conv_mem' + str(self.count)] = _tail_conv_mem
            spike_all['_tail_conv_spike' + str(self.count)] = _tail_conv_spike
        else:
            _tail_conv_mem, _tail_conv_spike = SpikeFunction(x, mem_all['_tail_conv_mem' + str(self.count)],
                                                                       spike_all['_tail_conv_spike' + str(self.count)], self.thresh)
            mem_all['_tail_conv_mem' + str(self.count)] = _tail_conv_mem
            spike_all['_tail_conv_spike' + str(self.count)] = _tail_conv_spike
        x = spike_all['_tail_conv_spike' + str(self.count)]
        num_of_spikes += tf.reduce_sum(x)
        tf.logging.info('tail conv outputs: %s shape: %s' % (x.name, x.shape))

    # shortcut connection in SRB
    if self._block_args0.id_skip:
      if all(s == 1 for s in self._block_args0.strides):
        x += inputs
    return x, num_of_spikes, num_of_synops

"""Class implements tf.keras.Model for the searched architecture found by NAS."""
class SNNModel(tf.keras.Model):
  # initializes a searched architecture found by NAS instance
  def __init__(self, dataset_name, override_params):
    super(SNNModel, self).__init__()
    num_classes = 10
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        num_classes=num_classes,
        depth_divisor=8,
        min_depth=None)
    if override_params:
        global_params = global_params._replace(**override_params)

    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    # threshold is trainable in our method
    # and its initial value is the result of its training during the search phase
    self.thresh = self.add_weight(shape=(1,),
                                 initializer=tf.random_normal_initializer(
                                     mean=thresh_last,
                                     stddev=0,
                                     seed=2022),
                                 name="thresh")
    self._build()

  # build the searched architecture
  def _build(self):
    self._blocks = []
    count = -1
    # Builds blocks.
    for block_args in self._blocks_args:
      if (block_args[0].num_repeat + block_args[1].num_repeat) > 0:
          count += 1
          block_args0 = block_args[0]
          block_args1 = block_args[1]
          self._blocks.append(TBS(block_args0, block_args1, self._global_params, self.thresh, count))
      else:
          continue

    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1

    # stem blocks
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        kernel_initializer=kernel_initializer['0'],
        padding='same',
        use_bias=False)
    self._bn0 = tf.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        beta_initializer=beta_initializer['0'],
        gamma_initializer=gamma_initializer['0'],
        fused=True)

    # tail blocks
    self._conv_head = tf.keras.layers.Conv2D(
        filters=1024,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn1 = tf.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        beta_initializer=conv_beta_initializer,
        gamma_initializer=conv_gamma_initializer,
        fused=True)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self._global_params.data_format)

    self._fc = tf.keras.layers.Dense(
        self._global_params.num_classes,
        kernel_initializer=dense_kernel_initializer,
        bias_initializer=dense_bias_initializer)

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
            outputs, num_of_spikes, num_of_synops = block.call(outputs, timestep, mem_all, spike_all, num_of_spikes, num_of_synops, training=training)

    # call tail blocks
    with tf.variable_scope('mnas_head'):
        num_of_synops += tf.reduce_sum(outputs * 1024 * tf.square(1/1))
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
