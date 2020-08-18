import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
rate = int(sys.argv[2])

input_fd = open(input_file, 'r')
vals = [0] * 1000
count = 0
for line in input_fd:
  strs = line.split(',')
  for i in range(len(strs)):
    vals[i] = vals[i] + float(strs[i])
  count = count + 1
  if count % rate == 0:
    output_str = ''
    for i in range(len(strs) - 1):
      output_str = output_str + ('%i,' % (vals[i] / rate))
      vals[i] = 0
    i = len(strs) - 1
    output_str = output_str + ('%f' % (vals[i] / rate))
    vals[i] = 0
    print output_str
