import torch

from cifar_vgg import VGG
from train import train_network

import math

def prune_network(args, network=None):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")

    if network is None:
        if args.data_set == 'CIFAR10':
            if 'vgg' in args.network:
                network = VGG(args.network, args.data_set)
        if args.load_path:
            check_point = torch.load(args.load_path)
            network.load_state_dict(check_point['state_dict'])

    # prune network
    if 'vgg' in args.network:
        network = global_pruning(network, args.prune_ratio)
        
    network = network.to(device)
    
    print("-*-"*10 + "\n\tPruned network\n" + "-*-"*10)
    if not args.retrain_flag:
        print(network)

    torch.save(network, './models/pruning_results/'+args.network+'_pruned_structure.pth')

    if args.retrain_flag:
        # update arguemtns for retraing pruned network
        args.epoch = args.retrain_epoch
        args.lr = args.retrain_lr
        args.lr_milestone = None # don't decay learning rate

        network = train_network(args, network)
    
    return network



def global_pruning(network, percentage):
    network = network.cpu()

    conv_count = 0 # conv count for 'indexing_prune_layers'
    dim = 0 # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]

    channel_group = []
    # get the channel index lists to be pruned
    channel_group = get_channel_index(network, percentage)
    
    print('Pruning:')
    # print out layers to be pruned and the number of filters
    for i in range(len(channel_group)):
        if len(channel_group[i]) > 0:
            if i > 12:
                print('layer {0}: {1} neurons'.format(i+1, len(channel_group[i])))
            else:
                print('layer {0}: {1} filters'.format(i+1, len(channel_group[i])))
    
    # go through all layers
    for i in range(len(network.features)):
        # if it's a convolutional layer
        if isinstance(network.features[i], torch.nn.Conv2d):            
            # need to change the input channel number 'cause the previous # of output channel has changed
            if dim == 1:
                new_ = get_new_conv(network.features[i], dim, channel_index)
                network.features[i] = new_
                dim ^= 1

            # if this conv layer has filters to be pruned
            if len(channel_group[conv_count]) > 0:
                channel_index = channel_group[conv_count]
                # get a new layer with pruned kernels
                new_ = get_new_conv(network.features[i], dim, channel_index)
                network.features[i] = new_

                # exclusive or 
                dim ^= 1
            
            conv_count += 1
        
        # if it's batchnorm layer and the previous conv layer has been modified
        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            # get new bn layer
            new_ = get_new_norm(network.features[i], 2, channel_index)
            network.features[i] = new_

    # if the # of output of the last conv layer has changed
    if len(channel_group[-2]) > 0:
        # modify the input of the fc layer
        network.classifier[0] = get_new_linear(network.classifier[0], 1, channel_index)
    
    # if the fc layer need pruning 
    if len(channel_group[-1]) > 0:
        # get channel index
        channel_index = channel_group[-1]

        # prune and modify the bn layer and the classification layer
        network.classifier[0] = get_new_linear(network.classifier[0], 0, channel_index)
        network.classifier[1] = get_new_norm(network.classifier[1], 1, channel_index)
        network.classifier[2] = get_new_linear(network.classifier[2], 1, channel_index)
    
    return network

# calculate the threshold and return the index list to be pruned
def perc(sorted_list, index_list, percentage):
    num = int(round(sorted_list.size(0) * percentage))

    return index_list[:num].tolist(), index_list[num]

def get_channel_index(network, percentage):
    count = 0 # counting sum of the accumulated channels
    stack = [] # stack of all L1-norm values
    accumulated_channels = [] # for calculating the index of each layer
    
    # loop to calculate the L1-norm of conv layers
    for i in range(len(network.features)):
        # if it's convolutional layer
        if isinstance(network.features[i], torch.nn.Conv2d):
            # get the filters
            filters = network.features[i].weight.data

            # counting the number of weights for all kernels
            _, in_channel, h, w = filters.size()
            num_weight = in_channel * h * w

            # calculate the L1-norm of each filter and normalize
            norm_of_filters = torch.sum(torch.abs(filters.view(filters.size(0), -1)), dim=1) / num_weight

            # counting the accumulated number of filters, for later calculating the index
            count += filters.size(0)
            accumulated_channels.append(count)

            # append to stack
            stack.append(norm_of_filters)

    # fc layer
    fc = network.classifier[0].weight.data
    norm_of_fc = torch.sum(torch.abs(fc), dim=1) / fc.size(1)
    stack.append(norm_of_fc)

    # add the number of fc layer neurons
    accumulated_channels.append(accumulated_channels[-1]+fc.size(1))

    # convert the stack to tensor
    stack = torch.cat(stack)

    # sort the stack
    vals, indices = torch.sort(stack)
    
    # get the list of indices to be pruned and the threshold
    channel_index, threshold = perc(vals, indices, percentage)

    # a list of index of all layers
    channel_group = [[] for i in range(len(accumulated_channels))]

    # judge each index belongs to which layer and append to the list
    for index in channel_index:
        # find out the index is in which layer
        for layer in range(len(accumulated_channels)):
            if index < accumulated_channels[layer]:
                if layer == 0: # the first layer
                    channel_group[layer].append(index)
                else:
                    channel_group[layer].append(index-accumulated_channels[layer-1])
                break

    return channel_group

def index_remove(tensor, dim, index):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    # new output channel size
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    # take out the index that need to be removed
    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    return new_tensor

def get_new_conv(conv, dim, channel_index):
    # dimension 0: output channel
    if dim == 0:
        # create an new convolution layer with the pruned number of output channels
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        if conv.bias is not None:
            new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv
    # dimension 1: input channel
    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_weight = index_remove(conv.weight.data, dim, channel_index)
        new_conv.weight.data = new_weight
        
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

        # return new_conv, residue
        return new_conv

def get_new_norm(norm, dim, channel_index):
    # create a new batchnormalization layer
    # for BN 2D
    if dim == 2:
        new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                        eps=norm.eps,
                                        momentum=norm.momentum,
                                        affine=norm.affine,
                                        track_running_stats=norm.track_running_stats)
    # for BN 1D
    elif dim == 1:
        new_norm = torch.nn.BatchNorm1d(num_features=int(norm.num_features - len(channel_index)),
                                        eps=norm.eps,
                                        momentum=norm.momentum,
                                        affine=norm.affine,
                                        track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)
        
    return new_norm

def get_new_linear(linear, dim, channel_index):
    # create a new fc layer
    if dim == 1:    # to modify the # of input channels
        new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                    out_features=linear.out_features,
                                    bias=linear.bias is not None)
        new_linear.weight.data = index_remove(linear.weight.data, dim, channel_index)
        new_linear.bias.data = linear.bias.data
    elif dim == 0:  # to modify the # of output channels
        new_linear = torch.nn.Linear(in_features=linear.in_features,
                                    out_features=int(linear.out_features - len(channel_index)),
                                    bias=linear.bias is not None)
        new_linear.weight.data = index_remove(linear.weight.data, dim, channel_index)
        new_linear.bias.data = index_remove(linear.bias.data, dim, channel_index)   # the bias also need to be pruned!!!
    
    return new_linear