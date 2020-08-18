import sys
import os
from datetime import datetime, timedelta

input_file_name = sys.argv[1]

input_file = open(input_file_name, 'r')
last_line = ''
for line in input_file:
  line = line.split('\n')[0]
  if line == '  type: "BatchNorm"':
    extra_line1 = last_line[:-1] + '/temp1"'
    extra_line2 = last_line[:-1] + '/temp2"'
    print extra_line1
    print extra_line2
  print line
  last_line = line
