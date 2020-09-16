# -*- coding: utf-8 -*-

def collect_params(model_list, exclude_bias_and_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    param_list = []
    for model in model_list:
        for name, param in model.named_parameters():
            if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
            else:
                param_dict = {'params': param}
            param_list.append(param_dict)
    return param_list
