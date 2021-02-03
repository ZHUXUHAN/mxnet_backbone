import mxnet as mx


def SeparableConv2d(data, in_channel, out_channel, kernel, stride, pad, no_bias=True, workspace=256, name=''):

    conv1 = mx.sym.Convolution(data=data, num_filter=in_channel, num_group=in_channel, kernel=kernel, stride=stride, \
                               pad=pad,  no_bias=no_bias, workspace=workspace, name=name + '_separableconv')

    # pointwise = mx.sym.Convolution(data=conv1, num_filter=out_channel, kernel=(1, 1), stride=(1, 1), \
    #                            pad=(0, 0),  no_bias=no_bias, workspace=workspace, name=name + '_pointwise')

    return conv1


def xx_block(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=256, memonger=False):
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')

    return act2 + shortcut


def bl_block(data, num_filter, stride, expansion_ratio, dim_match, name, bn_mom=0.9, workspace=256, memonger=False):

    conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * expansion_ratio), kernel=(1, 1), stride=(1, 1),
                               pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

    conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * expansion_ratio), kernel=(3, 3), stride=stride,
                               pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')

    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return act3 + shortcut


def dw_block(data, num_filter, stride, expansion_ratio, dim_match, name, bn_mom=0.9, workspace=256, memonger=False):
    conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * expansion_ratio), kernel=(1, 1), stride=(1, 1),
                               pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

    # conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * expansion_ratio), kernel=(3, 3), stride=stride,
    #                            pad=(1, 1), num_group=int(num_filter * expansion_ratio),
    #                            no_bias=True, workspace=workspace, name=name + '_conv2')

    conv2 = SeparableConv2d(data=act1, in_channel=int(num_filter * expansion_ratio), out_channel=int(num_filter * expansion_ratio), kernel=(3, 3), stride=stride,
                               pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')

    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

    # conv3 = SeparableConv2d(data=act2, in_channel=int(num_filter * expansion_ratio), out_channel=num_filter, kernel=(3, 3), stride=(1, 1), \
    #                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv3')

    # conv3 = SeparableConv2d(data=act2, in_channel=int(num_filter * expansion_ratio), out_channel=num_filter,
    #                         kernel=(3, 3), stride=(1, 1),
    #                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv3')

    # bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn3')
    # act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')

    # conv3 = SeparableConv2d(data=act2, in_channel=int(num_filter * expansion_ratio), out_channel=num_filter,
    #                         kernel=(3, 3), stride=(1, 1),
    #                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv3')
    #
    # bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    # act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')

    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')

    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return act3 + shortcut


def genet(units, num_stages, filter_list, num_classes, workspace=256, dtype='float32'):
    num_unit = len(units)
    assert (num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)

    conv_body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, name="conv0", workspace=workspace)
    conv_body = mx.sym.BatchNorm(data=conv_body, fix_gamma=False, eps=2e-5, momentum=0.9, name='conv0_bn1')
    conv_body = mx.sym.Activation(data=conv_body, act_type='relu', name='conv0_relu1')

    for i in range(num_stages):
        if i == 0:
            body = xx_block(conv_body, filter_list[i + 1], (2, 2), False, 'stage%d_unit%d' % (i + 1, 1))
        elif i == 1:
            body = xx_block(body, filter_list[i + 1], (2, 2), False,
                            'stage%d_unit%d' % (i + 1, 1))
            for j in range(units[i] - 1):
                body = xx_block(body, filter_list[i + 1], (1, 1), True,
                                'stage%d_unit%d' % (i + 1, j + 2))

        elif i == 2:
            body = bl_block(body, filter_list[i + 1], (2, 2), 0.25, False, 'stage%d_unit%d' % (i + 1, 1))
            for j in range(units[i] - 1):
                body = bl_block(body, filter_list[i + 1], (1, 1), 0.25, True, 'stage%d_unit%d' % (i + 1, j + 2))
        elif i == 3:
            body = dw_block(body, filter_list[i + 1], (2, 2), 3, False,
                            'stage%d_unit%d' % (i + 1, 1))
            for j in range(units[i] - 1):
                body = dw_block(body, filter_list[i + 1], (1, 1), 3, True,
                                'stage%d_unit%d' % (i + 1, j + 2))
        elif i == 4:
            body = dw_block(body, filter_list[i + 1], (1, 1), 3, False,
                            'stage%d_unit%d' % (i + 1, 1))
            for j in range(units[i] - 1):
                body = dw_block(body, filter_list[i + 1], (1, 1), 3, True,
                                'stage%d_unit%d' % (i + 1, j + 2))

    if dtype == 'float16':
        body = mx.sym.Cast(data=body, dtype=np.float32)  # for fp16

    body = mx.sym.Convolution(data=body, num_filter=filter_list[-1], kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                              no_bias=True, name="conv%d" % (num_stages + 1), workspace=workspace)

    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn1')
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')

    return fc1
    #


def get_symbol(num_classes, layer_type, conv_workspace=256, dtype='float32'):
    num_stages = 5
    if layer_type == 'light':
        units = [1, 3, 7, 2, 1]
        filter_list = [13, 48, 48, 384, 560, 256, 1920]
    elif layer_type == 'normal':
        units = [1, 2, 6, 4, 1]
        filter_list = [32, 128, 192, 640, 640, 640, 2560]
    elif layer_type == 'large':
        units = [1, 3, 7, 5, 4]
        filter_list = [32, 128, 192, 640, 640, 640, 2560]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return genet(units=units,
                 num_stages=num_stages,
                 filter_list=filter_list,
                 num_classes=num_classes,
                 workspace=conv_workspace,
                 dtype=dtype)


if __name__ == '__main__':
    data = mx.sym.Variable("data")
    symbol = get_symbol(1000, 'large')
    image_shape = (1, 3, 224, 224)
    mx.viz.print_summary(symbol, shape={"data":image_shape})
    mx.viz.plot_network(symbol, save_format='pdf', hide_weights=True, shape={"data":image_shape}).view()
