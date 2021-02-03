from symbol import fresnet
from symbol import resnet
from symbol import genet
import mxnet.ndarray as nd
import mxnet as mx
import json

from utils import calflops
from utils import calparams
from utils.mxnet_paser import mxnet_paser

# symbol = fresnet.get_symbol(1000, 100)
# symbol = resnet.get_symbol(1000, 50, '3, 224, 224')
symbol = genet.get_symbol(1000, 'large')
symbol.save("network_train.json")
# symbol = mx.symbol.load('/home/face_backbone/backbone/model-symbol.json')
image_shape = (1, 3, 224, 224)

paserer = mxnet_paser()
# mx.viz.print_summary(symbol, shape={"data":image_shape})

shape = {"data": image_shape}
interals = symbol.get_internals()
_, out_shapes, _ = interals.infer_shape(**shape)
if out_shapes is None:
    raise ValueError("Input shape is incomplete")
shape_dict = dict(zip(interals.list_outputs(), out_shapes))
conf = json.loads(symbol.tojson())
nodes = conf["nodes"]
heads = set(conf["heads"][0])
total_params = 0

for i, node in enumerate(nodes):
    total_params += calparams.calConvParams(node, nodes, shape_dict, heads)


ops = paserer.paser(symbol, 0, image_shape)

count_params = 0
total_ops = {
    'conv_multipflops': 0,
    'conv_addflops': 0,
    'conv_compareflops': 0,
    'conv_expflops': 0,
    'Fc_multipflops': 0,
    'Fc_addflops': 0,
    'Fc_compareflops': 0,
    'Fc_expflops': 0,
    'Pool_multipflops': 0,
    'Pool_addflops': 0,
    'Pool_compareflops': 0,
    'Pool_expflops': 0,
    'Activation_multipflops': 0,
    'Activation_addflops': 0,
    'Activation_compareflops': 0,
    'Activation_expflops': 0,
}

for op in ops:

    if op['type'] == 'Convolution':
        # print('kernel_size', op['attr']['kernel_size'])
        # print('in_shape', op['in_shape'])
        # print('out_shape', op['out_shape'])
        # print('has_bias', op['attr']["no_bias"])
        # print('num_group', op['attr']["num_group"])
        multipflops, addflops, compareflops, expflops = calflops.calConvFlops(op['in_shape'],
                                                                              op['out_shape'],
                                                                              op['attr']['kernel_size'],
                                                                              not op['attr']["no_bias"],
                                                                              op['attr']["num_group"])
        # print('conv', multipflops)
        total_ops['conv_multipflops'] += multipflops
        total_ops['conv_addflops'] += addflops
        total_ops['conv_compareflops'] += compareflops
        total_ops['conv_expflops'] += expflops

    elif op['type'] == 'FullyConnected':
        # print('in_shape', op['in_shape'])
        # print('out_shape', op['out_shape'])
        multipflops, addflops, compareflops, expflops = calflops.calFcFlops(op['in_shape'],
                                                                            op['out_shape'],
                                                                            not op['attr']["no_bias"])
        # total_ops['Fc_multipflops'] += multipflops
        # total_ops['Fc_addflops'] += addflops
        # total_ops['Fc_compareflops'] += compareflops
        # total_ops['Fc_expflops'] += expflops

    elif op['type'] == 'Pooling':
        # print('kernel_size', op['attr']['kernel_size'])
        # print('in_shape', op['in_shape'])
        # print('out_shape', op['out_shape'])
        # print('pool_type', op['attr']["pool_type"])
        try:
            multipflops, addflops, compareflops, expflops = calflops.calPoolingFlops(op['in_shape'],
                                                                                     op['out_shape'],
                                                                                     op['attr']['kernel_size'],
                                                                                     op['attr']["pool_type"])

            total_ops['Pool_multipflops'] += multipflops
            total_ops['Pool_addflops'] += addflops
            total_ops['Pool_compareflops'] += compareflops
            total_ops['Pool_expflops'] += expflops
        except:
            pass

    elif op['type'] == 'Activation':
        # print('in_shape', op['in_shape'])
        # print('out_shape', op['out_shape'])
        # print('act_type', op['attr']["act_type"])
        multipflops, addflops, compareflops, expflops = calflops.calActivationFlops(op['in_shape'],
                                                                                    op['out_shape'],
                                                                                    op['attr']["act_type"])
        total_ops['Activation_multipflops'] += multipflops
        total_ops['Activation_addflops'] += addflops
        total_ops['Activation_compareflops'] += compareflops
        total_ops['Activation_expflops'] += expflops
    else:
        pass

print('----total flops---')

total_flops = 0
for key in total_ops:
    total_flops+=total_ops[key] / 1e9
print('{}: {:.5f} GFlops'.format('Flops', total_flops))

for key in total_ops:
    print('{}: {:.5f} GFlops'.format(key, total_ops[key] / 1000000000.0))

print('----total params---')
print('{}: {:.5f} MParams'.format('Params', total_params / 1000000.0))
