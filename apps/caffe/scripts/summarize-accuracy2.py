import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
num_workers = 8
if len(sys.argv) > 2:
  num_workers = int(sys.argv[2])

input_fd = open(input_file, 'r')
times = []
accuracys = []
time_start = 0
for line in input_fd:
  strs = line.split()
  if time_start == 0 and (len(strs) == 9 or len(strs) == 10) and strs[6] == 'loss':
    time_tuple = datetime.strptime(strs[1], "%H:%M:%S.%f")
    time_start = time_tuple.hour * 3600 + time_tuple.minute * 60 + time_tuple.second
  if not len(strs) == 11:
    continue
  if not strs[8] == 'accuracy2' and not strs[8] == 'loss3/top-5':
    continue
  # times.append(float(strs[5][:-1]))
  time_tuple = datetime.strptime(strs[1], "%H:%M:%S.%f")
  times.append(time_tuple.hour * 3600 + time_tuple.minute * 60 + time_tuple.second)
  accuracys.append(float(strs[10]))

n = len(accuracys)
count = 0
for i in range(n):
  if times[i] < times[0]:
    times[i] = times[i] + 86400
  # print '%i,%i,%f'% (clocks[i], times[i] - times[0], losses[i])
  count = count + accuracys[i]
  if ((i + 1) % num_workers == 0):
    print '%i,%f'% (times[i] - time_start, count / num_workers)
    count = 0
