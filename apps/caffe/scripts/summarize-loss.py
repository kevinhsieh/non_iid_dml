import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
num_workers = 8
pdsh = 0
if len(sys.argv) > 2:
  num_workers = int(sys.argv[2])
if len(sys.argv) > 3:
  pdsh = int(sys.argv[3])

input_fd = open(input_file, 'r')
times = []
clocks = []
losses = []
for line in input_fd:
  strs = line.split()
  if not len(strs) == 9 + pdsh and not len(strs) == 10 + pdsh:
    continue
  if not strs[6 + pdsh] == 'loss':
    continue
  # times.append(float(strs[5][:-1]))
  time_tuple = datetime.strptime(strs[1 + pdsh], "%H:%M:%S.%f")
  times.append(time_tuple.hour * 3600 + time_tuple.minute * 60 + time_tuple.second)
  clocks.append(int(strs[5 + pdsh][:-1]))
  losses.append(float(strs[8 + pdsh]))

n = len(losses)
count = 0
for i in range(n):
  if times[i] < times[0]:
    times[i] = times[i] + 86400
  # print '%i,%i,%f'% (clocks[i], times[i] - times[0], losses[i])
  count = count + losses[i]
  if ((i + 1) % num_workers == 0):
    print '%i,%i,%f'% (clocks[i], times[i] - times[num_workers - 1], count / num_workers)
    count = 0
