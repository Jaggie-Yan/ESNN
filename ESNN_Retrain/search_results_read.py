import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""Read the model parameters obtained during the search phase."""
def checkpoint_read():
    kernel_all = []
    gamma_all = []
    beta_all = []
    fc_kernel = []
    fc_bias = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('./NAS_search_checkpoints')
            if ckpt and ckpt.model_checkpoint_path:
                reader = tf.train.NewCheckpointReader('./NAS_search_checkpoints/model.ckpt-40000')
                # all_variables = reader.get_variable_to_shape_map()
                # for variable_name in all_variables:
                #     print(variable_name, all_variables[variable_name])
                w = reader.get_tensor("single-path-search/single_path_super_net/mnas_stem/conv2d/kernel")
                kernel_all.append(w)
                gamma = reader.get_tensor("single-path-search/mnas_stem/batch_normalization/gamma")
                gamma_all.append(gamma)
                beta = reader.get_tensor("single-path-search/mnas_stem/batch_normalization/beta")
                beta_all.append(beta)

                w = reader.get_tensor('single-path-search/head_conv_kernel')
                kernel_all.append(w)
                gamma = reader.get_tensor("single-path-search/mnas_blocks_0/batch_normalization/gamma")
                gamma_all.append(gamma)
                beta = reader.get_tensor("single-path-search/mnas_blocks_0/batch_normalization/beta")
                beta_all.append(beta)

                w = reader.get_tensor('single-path-search/tial_conv_kernel')
                kernel_all.append(w)
                gamma = reader.get_tensor("single-path-search/mnas_blocks_0/batch_normalization_1/gamma")
                gamma_all.append(gamma)
                beta = reader.get_tensor("single-path-search/mnas_blocks_0/batch_normalization_1/beta")
                beta_all.append(beta)

                for i in range(9): # 9: the total number of TBS blocks on CIFAR10 dataset
                    w = reader.get_tensor('single-path-search/head_conv_kernel_' + str(i+1))
                    kernel_all.append(w)
                    gamma = reader.get_tensor('single-path-search/mnas_blocks_' + str(i+1) + '/batch_normalization/gamma')
                    gamma_all.append(gamma)
                    beta = reader.get_tensor('single-path-search/mnas_blocks_' + str(i+1) + '/batch_normalization/beta')
                    beta_all.append(beta)

                    w = reader.get_tensor('single-path-search/tial_conv_kernel_' + str(i+1))
                    kernel_all.append(w)
                    gamma = reader.get_tensor('single-path-search/mnas_blocks_' + str(i + 1) + '/batch_normalization_1/gamma')
                    gamma_all.append(gamma)
                    beta = reader.get_tensor('single-path-search/mnas_blocks_' + str(i + 1) + '/batch_normalization_1/beta')
                    beta_all.append(beta)

                w = reader.get_tensor("single-path-search/single_path_super_net/mnas_head/conv2d_1/kernel")
                kernel_all.append(w)
                gamma = reader.get_tensor("single-path-search/mnas_head/batch_normalization/gamma")
                gamma_all.append(gamma)
                beta = reader.get_tensor("single-path-search/mnas_head/batch_normalization/beta")
                beta_all.append(beta)

                w = reader.get_tensor("single-path-search/single_path_super_net/mnas_head/dense/kernel")
                fc_kernel.append(w)
                bias = reader.get_tensor("single-path-search/single_path_super_net/mnas_head/dense/bias")
                fc_bias.append(bias)
            else:
                print('No checkpoint file found')
    return kernel_all, gamma_all, beta_all, fc_kernel, fc_bias