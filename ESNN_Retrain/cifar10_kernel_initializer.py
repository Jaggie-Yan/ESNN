import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from search_results_read import checkpoint_read
from search_model_read import parse_netarch_model

blocks_args_init, block_inds_last, thresh_last = parse_netarch_model('./NAS_search_results/')
blocks_args = []
for i in range(int(len(blocks_args_init)/2)):
    temp = []
    temp.append(blocks_args_init[i*2])
    temp.append(blocks_args_init[i*2+1])
    blocks_args.append(temp)

kernel_all, gamma_all, beta_all, fc_kernel, fc_bias = checkpoint_read()
kernel_initializer_all = {}
gamma_initializer_all = {}
beta_initializer_all = {}

"""Obtain the selected kernel parameters according to the searched kernel size results."""
def conv_kernel_initializer0(shape, dtype=None, partition_info=None):
  del partition_info
  return tf.cast(kernel_all[0], tf.float32)
kernel_initializer_all['0'] = conv_kernel_initializer0

def conv_kernel_initializer1(shape, dtype=None, partition_info=None):
  del partition_info
  return tf.cast(kernel_all[1], tf.float32)
kernel_initializer_all['1'] = conv_kernel_initializer1

def conv_kernel_initializer11(shape, dtype=None, partition_info=None):
  del partition_info
  return tf.cast(kernel_all[2], tf.float32)
kernel_initializer_all['11'] = conv_kernel_initializer11

def conv_kernel_initializer2(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[0][0] == 0:
      search_w = 0
  else:
      if block_inds_last[0][1] == 0:
          search_w = kernel_all[3][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[3]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['2'] = conv_kernel_initializer2

def conv_kernel_initializer21(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[0][2] == 0:
      search_w = 0
  else:
      if block_inds_last[0][3] == 0:
          search_w = kernel_all[4][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[4]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['21'] = conv_kernel_initializer21

def conv_kernel_initializer3(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[1][0] == 0:
      search_w = 0
  else:
      if block_inds_last[1][1] == 0:
          search_w = kernel_all[5][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[5]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['3'] = conv_kernel_initializer3

def conv_kernel_initializer31(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[1][2] == 0:
      search_w = 0
  else:
      if block_inds_last[1][3] == 0:
          search_w = kernel_all[6][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[6]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['31'] = conv_kernel_initializer31

def conv_kernel_initializer4(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[2][0] == 0:
      search_w = 0
  else:
      if block_inds_last[2][1] == 0:
          search_w = kernel_all[7][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[7]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['4'] = conv_kernel_initializer4

def conv_kernel_initializer41(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[2][2] == 0:
      search_w = 0
  else:
      if block_inds_last[2][3] == 0:
          search_w = kernel_all[8][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[8]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['41'] = conv_kernel_initializer41

def conv_kernel_initializer5(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[3][0] == 0:
      search_w = 0
  else:
      if block_inds_last[3][1] == 0:
          search_w = kernel_all[9][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[9]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['5'] = conv_kernel_initializer5

def conv_kernel_initializer51(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[3][2] == 0:
      search_w = 0
  else:
      if block_inds_last[3][3] == 0:
          search_w = kernel_all[10][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[10]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['51'] = conv_kernel_initializer51

def conv_kernel_initializer6(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[4][0] == 0:
      search_w = 0
  else:
      if block_inds_last[4][1] == 0:
          search_w = kernel_all[11][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[11]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['6'] = conv_kernel_initializer6

def conv_kernel_initializer61(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[4][2] == 0:
      search_w = 0
  else:
      if block_inds_last[4][3] == 0:
          search_w = kernel_all[12][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[12]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['61'] = conv_kernel_initializer61

def conv_kernel_initializer7(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[5][0] == 0:
      search_w = 0
  else:
      if block_inds_last[5][1] == 0:
          search_w = kernel_all[13][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[13]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['7'] = conv_kernel_initializer7

def conv_kernel_initializer71(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[5][2] == 0:
      search_w = 0
  else:
      if block_inds_last[5][3] == 0:
          search_w = kernel_all[14][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[14]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['71'] = conv_kernel_initializer71

def conv_kernel_initializer8(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[6][0] == 0:
      search_w = 0
  else:
      if block_inds_last[6][1] == 0:
          search_w = kernel_all[15][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[15]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['8'] = conv_kernel_initializer8

def conv_kernel_initializer81(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[6][2] == 0:
      search_w = 0
  else:
      if block_inds_last[6][3] == 0:
          search_w = kernel_all[16][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[16]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['81'] = conv_kernel_initializer81

def conv_kernel_initializer9(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[7][0] == 0:
      search_w = 0
  else:
      if block_inds_last[7][1] == 0:
          search_w = kernel_all[17][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[17]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['9'] = conv_kernel_initializer9

def conv_kernel_initializer91(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[7][2] == 0:
      search_w = 0
  else:
      if block_inds_last[7][3] == 0:
          search_w = kernel_all[18][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[18]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['91'] = conv_kernel_initializer91

def conv_kernel_initializer10(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[8][0] == 0:
      search_w = 0
  else:
      if block_inds_last[8][1] == 0:
          search_w = kernel_all[19][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[19]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['10'] = conv_kernel_initializer10

def conv_kernel_initializer101(shape, dtype=None, partition_info=None):
  del partition_info
  if block_inds_last[8][2] == 0:
      search_w = 0
  else:
      if block_inds_last[8][3] == 0:
          search_w = kernel_all[20][1:4, 1:4, :, :]
      else:
          search_w = kernel_all[20]
  return tf.cast(search_w, tf.float32)
kernel_initializer_all['101'] = conv_kernel_initializer101

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  del partition_info
  return tf.cast(kernel_all[-1], tf.float32)

def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  del partition_info
  return tf.cast(fc_kernel[0], tf.float32)

"""Get the bata and gamma parameters obtained during the search phase."""
def gamma_initializer0(shape, dtype=tf.float32):
    return tf.cast(gamma_all[0], tf.float32)
gamma_initializer_all['0'] = gamma_initializer0

def beta_initializer0(shape, dtype=tf.float32):
    return tf.cast(beta_all[0], tf.float32)
beta_initializer_all['0'] = beta_initializer0

def gamma_initializer1(shape, dtype=tf.float32):
    return tf.cast(gamma_all[1], tf.float32)
gamma_initializer_all['1'] = gamma_initializer1

def beta_initializer1(shape, dtype=tf.float32):
    return tf.cast(beta_all[1], tf.float32)
beta_initializer_all['1'] = beta_initializer1

def gamma_initializer11(shape, dtype=tf.float32):
    return tf.cast(gamma_all[2], tf.float32)
gamma_initializer_all['11'] = gamma_initializer11

def beta_initializer11(shape, dtype=tf.float32):
    return tf.cast(beta_all[2], tf.float32)
beta_initializer_all['11'] = beta_initializer11

def gamma_initializer2(shape, dtype=tf.float32):
    return tf.cast(gamma_all[3], tf.float32)
gamma_initializer_all['2'] = gamma_initializer2

def beta_initializer2(shape, dtype=tf.float32):
    return tf.cast(beta_all[3], tf.float32)
beta_initializer_all['2'] = beta_initializer2

def gamma_initializer21(shape, dtype=tf.float32):
    return tf.cast(gamma_all[4], tf.float32)
gamma_initializer_all['21'] = gamma_initializer21

def beta_initializer21(shape, dtype=tf.float32):
    return tf.cast(beta_all[4], tf.float32)
beta_initializer_all['21'] = beta_initializer21

def gamma_initializer3(shape, dtype=tf.float32):
    return tf.cast(gamma_all[5], tf.float32)
gamma_initializer_all['3'] = gamma_initializer3

def beta_initializer3(shape, dtype=tf.float32):
    return tf.cast(beta_all[5], tf.float32)
beta_initializer_all['3'] = beta_initializer3

def gamma_initializer31(shape, dtype=tf.float32):
    return tf.cast(gamma_all[6], tf.float32)
gamma_initializer_all['31'] = gamma_initializer31

def beta_initializer31(shape, dtype=tf.float32):
    return tf.cast(beta_all[6], tf.float32)
beta_initializer_all['31'] = beta_initializer31

def gamma_initializer4(shape, dtype=tf.float32):
    return tf.cast(gamma_all[7], tf.float32)
gamma_initializer_all['4'] = gamma_initializer4

def beta_initializer4(shape, dtype=tf.float32):
    return tf.cast(beta_all[7], tf.float32)
beta_initializer_all['4'] = beta_initializer4

def gamma_initializer41(shape, dtype=tf.float32):
    return tf.cast(gamma_all[8], tf.float32)
gamma_initializer_all['41'] = gamma_initializer41

def beta_initializer41(shape, dtype=tf.float32):
    return tf.cast(beta_all[8], tf.float32)
beta_initializer_all['41'] = beta_initializer41

def gamma_initializer5(shape, dtype=tf.float32):
    return tf.cast(gamma_all[9], tf.float32)
gamma_initializer_all['5'] = gamma_initializer5

def beta_initializer5(shape, dtype=tf.float32):
    return tf.cast(beta_all[9], tf.float32)
beta_initializer_all['5'] = beta_initializer5

def gamma_initializer51(shape, dtype=tf.float32):
    return tf.cast(gamma_all[10], tf.float32)
gamma_initializer_all['51'] = gamma_initializer51

def beta_initializer51(shape, dtype=tf.float32):
    return tf.cast(beta_all[10], tf.float32)
beta_initializer_all['51'] = beta_initializer51

def gamma_initializer6(shape, dtype=tf.float32):
    return tf.cast(gamma_all[11], tf.float32)
gamma_initializer_all['6'] = gamma_initializer6

def beta_initializer6(shape, dtype=tf.float32):
    return tf.cast(beta_all[11], tf.float32)
beta_initializer_all['6'] = beta_initializer6

def gamma_initializer61(shape, dtype=tf.float32):
    return tf.cast(gamma_all[12], tf.float32)
gamma_initializer_all['61'] = gamma_initializer61

def beta_initializer61(shape, dtype=tf.float32):
    return tf.cast(beta_all[12], tf.float32)
beta_initializer_all['61'] = beta_initializer61

def gamma_initializer7(shape, dtype=tf.float32):
    return tf.cast(gamma_all[13], tf.float32)
gamma_initializer_all['7'] = gamma_initializer7

def beta_initializer7(shape, dtype=tf.float32):
    return tf.cast(beta_all[13], tf.float32)
beta_initializer_all['7'] = beta_initializer7

def gamma_initializer71(shape, dtype=tf.float32):
    return tf.cast(gamma_all[14], tf.float32)
gamma_initializer_all['71'] = gamma_initializer71

def beta_initializer71(shape, dtype=tf.float32):
    return tf.cast(beta_all[14], tf.float32)
beta_initializer_all['71'] = beta_initializer71

def gamma_initializer8(shape, dtype=tf.float32):
    return tf.cast(gamma_all[15], tf.float32)
gamma_initializer_all['8'] = gamma_initializer8

def beta_initializer8(shape, dtype=tf.float32):
    return tf.cast(beta_all[15], tf.float32)
beta_initializer_all['8'] = beta_initializer8

def gamma_initializer81(shape, dtype=tf.float32):
    return tf.cast(gamma_all[16], tf.float32)
gamma_initializer_all['81'] = gamma_initializer81

def beta_initializer81(shape, dtype=tf.float32):
    return tf.cast(beta_all[16], tf.float32)
beta_initializer_all['81'] = beta_initializer81

def gamma_initializer9(shape, dtype=tf.float32):
    return tf.cast(gamma_all[17], tf.float32)
gamma_initializer_all['9'] = gamma_initializer9

def beta_initializer9(shape, dtype=tf.float32):
    return tf.cast(beta_all[17], tf.float32)
beta_initializer_all['9'] = beta_initializer9

def gamma_initializer91(shape, dtype=tf.float32):
    return tf.cast(gamma_all[18], tf.float32)
gamma_initializer_all['91'] = gamma_initializer91

def beta_initializer91(shape, dtype=tf.float32):
    return tf.cast(beta_all[18], tf.float32)
beta_initializer_all['91'] = beta_initializer91

def gamma_initializer10(shape, dtype=tf.float32):
    return tf.cast(gamma_all[19], tf.float32)
gamma_initializer_all['10'] = gamma_initializer10

def beta_initializer10(shape, dtype=tf.float32):
    return tf.cast(beta_all[19], tf.float32)
beta_initializer_all['10'] = beta_initializer10

def gamma_initializer101(shape, dtype=tf.float32):
    return tf.cast(gamma_all[20], tf.float32)
gamma_initializer_all['101'] = gamma_initializer101

def beta_initializer101(shape, dtype=tf.float32):
    return tf.cast(beta_all[20], tf.float32)
beta_initializer_all['101'] = beta_initializer101

def conv_gamma_initializer(shape, dtype=tf.float32):
    return tf.cast(gamma_all[-1], tf.float32)

def conv_beta_initializer(shape, dtype=tf.float32):
    return tf.cast(beta_all[-1], tf.float32)

def dense_bias_initializer(shape, dtype=tf.float32):
    return tf.cast(fc_bias[-1], tf.float32)