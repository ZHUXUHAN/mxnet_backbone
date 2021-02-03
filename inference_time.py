import time
import numpy as np
import mxnet as mx
from symbol import fresnet
from symbol import resnet
from symbol import genet
from collections import namedtuple


def single_input():
    img = np.zeros((4, 3, 112, 112))
    array = mx.nd.array(img)
    return array


prefix = '/home/face/zhuxuhan/model_zoo/varresnet/model_fp16'
epoch = 0
time0 = time.time()
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
print(sym)

# 提取中间某层输出帖子特征层作为输出
# all_layers = sym.get_internals()
# sym = all_layers['fc1_output']
layer = 'fc1'
all_layers = sym.get_internals()
sym = all_layers[layer + "_output"]

# 重建模型
model = mx.mod.Module(symbol=all_layers, label_names=None, context=mx.gpu(0))
model.bind(for_training=False, data_shapes=[('data', (4, 3, 112, 112))])
model.set_params(arg_params, aux_params)

time1 = time.time()

time_load = time1 - time0
print("模型加载和重建时间：{0}".format(time_load))

Batch = namedtuple("batch", ['data'])
array = single_input()
for i in range(100):
    model.forward(Batch([array]), is_train=False)
    vector = model.get_outputs()[0].asnumpy()
mx.nd.waitall()

time_inference = time.time()
time_frame = time_inference - time1

print("模型前向时间：{0}".format(time_frame))
