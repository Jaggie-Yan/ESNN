"""Parsing the NAS search progress."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# filter number setting according to the search space
ichannels = ['_i128', '_i128', '_i256']
inner_channels = ['_i128', '_i256', '_i512']
ochannels = ['_o128', '_o256', '_o512']

# group number and the repeat number of blocks in each group setting according to the search space
stage_num = 3
inner_blocks = 3
repeat_num = stage_num * inner_blocks

# architecture parameters setting according to the search space
stride2_layers = [0, 3, 6]  # these you cannot drop
blocks_first = ['r1_k3_s11_i64_o128_noskip', 'r1_k3_s11_i128_o128_noskip']
labels_num = 5

"""Get the detailed results of the searched architecture."""
def parse_indicators_single_path_nas(path, tf_size_guidance):
    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show the needed tags in the log file
    labels = ['d3x3_head_', 'd5x5_head_', 'd3x3_tail_', 'd5x5_tail_', 'thresh_']
    inds = []
    for idx in range(repeat_num):
        layer_row = []
        for cnt, label_ in enumerate(labels):
            summary_label_ = label_ + str(idx + 1)
            decision_ij = event_acc.Scalars(summary_label_)
            element = []
            if cnt < 4:
                for num in range(len(decision_ij)):
                    if decision_ij[num].value >= 1.0:
                        value = 1
                    else:
                        value = 0
                    element.append(value)
            else:
                for num in range(len(decision_ij)):
                    element.append(decision_ij[num].value)
            layer_row.append(element)
        inds.append(layer_row)
    return inds, len(decision_ij)

"""Get the searched layer types."""
def encode_single_path_nas_arch(inds, LEN_NUM):
    print('Sampling network')
    network_all = []
    for len_num in range(LEN_NUM):
        network = []
        candidate_ops = ['3x3-', '5x5-', 'skip']
        for layer_cnt in range(repeat_num):
            temp = []
            inds_row = [inds[layer_cnt][0][len_num], inds[layer_cnt][1][len_num]]
            if inds_row == [0.0, 0.0]:
                idx = 2
            elif inds_row == [0.0, 1.0]:
                idx = 2
            elif inds_row == [1.0, 0.0]:
                idx = 0
            elif inds_row == [1.0, 1.0]:
                idx = 1
            else:
                print(inds_row)
                assert 0 == 1  # will crash
            temp.append(candidate_ops[idx])

            inds_row = [inds[layer_cnt][2][len_num], inds[layer_cnt][3][len_num]]

            if inds_row == [0.0, 0.0]:
                idx = 2
            elif inds_row == [0.0, 1.0]:
                idx = 2
            elif inds_row == [1.0, 0.0]:
                idx = 0
            elif inds_row == [1.0, 1.0]:
                idx = 1
            else:
                print(inds_row)
                assert 0 == 1  # will crash
            temp.append(candidate_ops[idx])
            network.append(temp)
        network_all.append(network)
    block_inds_last = []
    for layer_cnt in range(repeat_num):
        inds_row_last = [inds[layer_cnt][0][-1], inds[layer_cnt][1][-1], inds[layer_cnt][2][-1], inds[layer_cnt][3][-1]]
        block_inds_last.append(inds_row_last)
    return network_all, block_inds_last, inds[-1][4][-1]

"""Print the searched architecture."""
def print_net(network):
    for idx, layer in enumerate(network[-1]):
        print('search_block_' + str(idx + 1))
        for idx1, layer1 in enumerate(layer):
            print(idx1, layer1)

"""Encode the searched architecture."""
def convnet_encoder(network):
    blocks_args_all = []
    for net_num in range(len(network)):
        # this encodes our layer types to the needed encoding for the model generation!
        block_cnt = 0
        blocks_args = []
        blocks_args.append(blocks_first[0])
        blocks_args.append(blocks_first[1])
        for stage_idx in range(stage_num):
            for inner_block in range(inner_blocks):
                for conv_idx in range(2):
                    layer_type = network[net_num][block_cnt][conv_idx]
                    if layer_type == 'skip':
                        if conv_idx == 0:
                            assert block_cnt not in stride2_layers
                        next_block_encoding = 'r0_' + 'k0' + '_s00' + '_i0' + '_o0'
                        blocks_args.append(next_block_encoding)
                    else:
                        if layer_type == '3x3-':
                            kernel_sample = 'k3'
                        elif layer_type == '5x5-':
                            kernel_sample = 'k5'

                        # bug found! 1st block of each group does not drop!
                        if block_cnt in stride2_layers and conv_idx == 0:
                            stride_sample = '_s22'
                        else:
                            stride_sample = '_s11'

                        if inner_block == 0 and conv_idx == 0:
                            ich_ = ichannels[stage_idx]
                        else:
                            ich_ = inner_channels[stage_idx]

                        next_block_encoding = 'r1_' + kernel_sample + stride_sample + ich_ + ochannels[stage_idx]
                        blocks_args.append(next_block_encoding)
                block_cnt += 1
        blocks_args_all.append(blocks_args)
    return blocks_args_all







