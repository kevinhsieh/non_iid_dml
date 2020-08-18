import sys
import os
from string import maketrans 

file_name = sys.argv[1]
num_dups = int(sys.argv[2])
intab = '%'
if len(sys.argv) > 3:
  intab = sys.argv[3]
pattern = ''
if len(sys.argv) > 4:
  pattern = sys.argv[4]

template_file = '%s.template' % (file_name)
template_fd = open(template_file, 'r')

output_files = []
output_fds = []
for i in range(num_dups):
  output_file = '%s.%i' % (file_name, i)
  if pattern != '':
    output_file = pattern % i
  output_files.append(output_file)
  output_fds.append(open(output_file, 'w'))

for line in template_fd:
  for i in range(num_dups):
    outtab = str(i)
    # trantab = maketrans(intab, outtab)
    # print line.translate(trantab)
    # output_fds[i].write(line.translate(trantab))
    output_fds[i].write(line.replace(intab, outtab))

template_fd.close()
for i in range(num_dups):
  output_fds[i].close()

for output_file in output_files:
  os.system('chmod a+x %s' % output_file)
