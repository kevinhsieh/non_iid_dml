#!/bin/bash

pdsh -R ssh -w ^examples/cifar10/10parts/machinefile "pkill caffe_geeps"

LOG=output.txt
OUTDIR=
NET=${1}

if [ "$#" -eq 2 ]; then
  mkdir $2
  pwd > $2/pwd
  SED_STR='s#snapshot_prefix: "[a-zA-Z0-9/_.,]*"#snapshot_prefix: "'"${2}"'/cifar_snapshot"#g'
  sed -i ''"${SED_STR}"'' ./examples/cifar10/10parts/${NET}_solver.prototxt.template
  cp examples/cifar10/10parts/train_net.sh $2/.
  cp examples/cifar10/10parts/${NET}_train_val.prototxt.template $2/.
  cp examples/cifar10/10parts/${NET}_solver.prototxt.template $2/.
  cp examples/cifar10/10parts/machinefile $2/.
  cp examples/cifar10/10parts/ps_config_${NET} $2/.
  LOG=$2/output.txt
  OUTDIR=$2
fi

echo '' > ${LOG}

python scripts/duplicate.py examples/cifar10/10parts/${NET}_train_val.prototxt 10
python scripts/duplicate.py examples/cifar10/10parts/${NET}_solver.prototxt 10

# Copy files to all the other worker nodes, only do this when there is no
# shared file system among worker nodes
# Note: The first line in the machinefile must be the master node!
readarray -t nodes < examples/cifar10/10parts/machinefile

for n in "${nodes[@]:1}"
do
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/cifar10/10parts/machinefile ${n}:$(pwd)/examples/cifar10/10parts/
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/cifar10/10parts/ps_config_${NET} ${n}:$(pwd)/examples/cifar10/10parts/
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/cifar10/10parts/${NET}_train_val.prototxt.* ${n}:$(pwd)/examples/cifar10/10parts/
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/cifar10/10parts/${NET}_solver.prototxt.* ${n}:$(pwd)/examples/cifar10/10parts/
done

pdsh -R ssh -w ^examples/cifar10/10parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/10parts/${NET}_solver.prototxt --ps_config=examples/cifar10/10parts/ps_config_${NET} --machinefile=examples/cifar10/10parts/machinefile --worker_id=%n --outdir=${OUTDIR}" 2>&1 | tee ${LOG}
