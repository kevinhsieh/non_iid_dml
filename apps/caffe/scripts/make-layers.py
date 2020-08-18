import sys
import os
from string import maketrans 

num_layers = 64
for i in range(num_layers):
  print('layer {')
  print('  name: "fc%i"' % (i + 1))
  print('  type: "InnerProduct"')
  print('  bottom: "fc%i"' % i)
  print('  top: "fc%i"' % (i + 1))
  print('  param {')
  print('    lr_mult: 1')
  print('    decay_mult: 1')
  print('  }')
  print('  param {')
  print('    lr_mult: 2')
  print('    decay_mult: 0')
  print('  }')
  print('  inner_product_param {')
  print('    num_output: 10000')
  # print('    weight_filler {')
  # print('      type: "gaussian"')
  # print('      std: 0.005')
  # print('    }')
  # print('    bias_filler {')
  # print('      type: "constant"')
  # print('      value: 0.1')
  # print('    }')
  print('  }')
  print('}')
