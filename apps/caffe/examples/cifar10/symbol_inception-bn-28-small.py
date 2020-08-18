"""
simplified inception-bn.py for images has size around 28 x 28
"""

import mxnet_converter as mxc

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu", mirror_attr={}, name=None, suffix=''):
    conv = mxc.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mxc.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mxc.Activation(data = bn, act_type=act_type, name='relu_%s%s' %(name, suffix))
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3, name):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1), name='%s_3x3' % name)
    # pool
    pool = mxc.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', name='max_pool_%s_pool' % (name))
    # concat
    concat = mxc.Concat(bottoms=[conv, pool], name='ch_concat_%s_chconcat' % name)
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, name):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1, name='%s_1x1' % name)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3, name='%s_3x3' % name)
    #concat
    concat = mxc.Concat(bottoms=[conv1x1, conv3x3], name='ch_concat_%s_chconcat' % name)
    return concat

num_classes = 10
force_mirroring=False
if force_mirroring:
    attr = {'force_mirroring': 'true'}
else:
    attr = {}

# data = mxc.Variable(name="data")
data = 'data'
conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu", name='conv1')
in3a = SimpleFactory(conv1, 32, 32, name='in3a')
in3b = SimpleFactory(in3a, 32, 48, name='in3b')
in3c = DownsampleFactory(in3b, 80, name='in3c')
in4a = SimpleFactory(in3c, 112, 48, name='in4a')
in4b = SimpleFactory(in4a, 96, 64, name='in4b')
in4c = SimpleFactory(in4b, 80, 80, name='in4c')
in4d = SimpleFactory(in4c, 48, 96, name='in4d')
in4e = DownsampleFactory(in4d, 96, name='in4e')
in5a = SimpleFactory(in4e, 176, 160, name='in5a')
in5b = SimpleFactory(in5a, 176, 160, name='in5b')
pool = mxc.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool")
flatten = mxc.Flatten(data=pool, name="flatten1")
fc = mxc.FullyConnected(data=flatten, num_hidden=num_classes, name="fc1")
# softmax = mxc.SoftmaxOutput(data=fc, name="softmax")
