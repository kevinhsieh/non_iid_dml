def Convolution(data, name, num_filter, kernel, stride=[1,1], pad=[0,0]):
  bottom = data
  top = name
  print 'layer {'
  print '  bottom: "%s"' % bottom
  print '  top: "%s"' % top
  print '  name: "%s"' % name
  print '  type: "Convolution"'
  print '  convolution_param {'
  print '    num_output: %i' % num_filter
  if pad[0] != 0:
    print '    pad: %i' % pad[0]
  print '    kernel_size: %i' % kernel[0]
  print '    stride: %i' % stride[0]
  print '    weight_filler {'
  print '      type: "xavier"'
  print '    }'
  print '    bias_term: false'
  print '  }'
  print '}'
  return top

def BatchNorm(data, name):
  bottom = data
  top = name
  temp1 = top + '/temp1'
  temp2 = top + '/temp2'
  print 'layer {'
  print '  bottom: "%s"' % bottom
  print '  name: "%s"' % name
  print '  top: "%s"' % top
  print '  top: "%s"' % temp1
  print '  top: "%s"' % temp2
  print '  type: "BatchNorm"'
  print '}'
  bottom2 = top
  name2 = bottom2 + '/sc'
  top2 = name2
  print 'layer {'
  print '  bottom: "%s"' % bottom2
  print '  top: "%s"' % top2
  print '  name: "%s"' % name2
  print '  type: "Scale"'
  print '  scale_param {'
  print '    bias_term: true'
  print '  }'
  print '}'
  return top2

def Activation(data, name, act_type):
  assert act_type == "relu"
  bottom = data
  top = bottom
  print 'layer {'
  print '  bottom: "%s"' % bottom
  print '  top: "%s"' % top
  print '  name: "%s"' % name
  print '  type: "ReLU"'
  print '}'
  return top

def Pooling(data, name, kernel, stride=[1,1], pad=[0,0], pool_type='max'):
  if pool_type == 'max':
    return MaxPooling(data, name, kernel, stride, pad, pool_type)
  else:
    return AvePooling(data, name, kernel, stride, pad, pool_type)

def MaxPooling(data, name, kernel, stride, pad, pool_type):
  bottom = data
  top = name
  temp = top + '/temp'
  print 'layer {'
  print '  bottom: "%s"' % bottom
  print '  top: "%s"' % top
  print '  top: "%s"' % temp
  print '  name: "%s"' % name
  print '  type: "Pooling"'
  print '  pooling_param {'
  print '    pool: MAX'
  print '    kernel_size: %i' % kernel[0]
  if pad[0] != 0:
    print '    pad: %i' % pad[0]
  print '    stride: %i' % stride[0]
  print '  }'
  print '}'
  return top

def AvePooling(data, name, kernel, stride, pad, pool_type):
  bottom = data
  top = name
  print 'layer {'
  print '  bottom: "%s"' % bottom
  print '  top: "%s"' % top
  print '  name: "%s"' % name
  print '  type: "Pooling"'
  print '  pooling_param {'
  print '    pool: AVE'
  print '    kernel_size: %i' % kernel[0]
  if pad[0] != 0:
    print '    pad: %i' % pad[0]
  print '    stride: %i' % stride[0]
  print '  }'
  print '}'
  return top

def Concat(bottoms, name):
  top = name
  print 'layer {'
  for bottom in bottoms:
    print '  bottom: "%s"' % bottom
  print '  top: "%s"' % top
  print '  name: "%s"' % name
  print '  type: "Concat"'
  print '}'
  return top

def FullyConnected(data, name, num_hidden):
  bottom = data
  top = name
  print 'layer {'
  print '  bottom: "%s"' % bottom
  print '  top: "%s"' % top
  print '  name: "%s"' % name
  print '  type: "InnerProduct"'
  print '  inner_product_param {'
  print '    num_output: %i' % num_hidden
  print '    weight_filler {'
  print '      type: "xavier"'
  print '    }'
  print '    bias_filler {'
  print '      type: "constant"'
  print '      value: 0'
  print '    }'
  print '  }'
  print '}'
  return top

def Flatten(data, name):
  bottom = data
  top = bottom
  return top
