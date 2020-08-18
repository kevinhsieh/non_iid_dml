import sys
import caffe
import lmdb
from collections import defaultdict

lmdb_env = lmdb.open(sys.argv[1])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

label_count = defaultdict(int)
count = 0

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    label_count[label] += 1
    count += 1

print(label_count)
print(count)
