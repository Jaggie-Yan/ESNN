import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model_def

def build_model(images, dataset_name, model_name, training, override_params=None,
        parse_search_dir=None):
  """A helper function to create the searched architecture found by NAS and return predicted logits and the number of spikes and SynOps.

  Args:
    images: input images tensor.
    dataset_name: string, the dataset name.
    model_name: string, the model name.
    training: boolean, whether the model is constructed for training.

  Returns:
    logits: the logits tensor of classes.
    num_of_spikes: the number of total spikes based on the threshold decisions.
    num_of_synops: the number of total SynOps based on the threshold decisions.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  if model_name == 'single-path':
    assert parse_search_dir is not None
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  with tf.variable_scope(model_name):
    model = model_def.SNNModel(dataset_name, override_params)
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
            logits = _logits
        else:
            logits += _logits

    logits = logits / TimeStep
  logits = tf.identity(logits, 'logits')
  return logits, num_of_spikes, num_of_synops
