#!/bin/bash

pdsh -R ssh -w ^examples/geoanimal/13parts/machinefile "pkill caffe_geeps"

LOG=output.txt
OUTDIR=
NET=googlenetv2

if [ "$#" -eq 1 ]; then
  mkdir -p $1
  pwd > $1/pwd
  #git status > $1/git-status
  #git diff > $1/git-diff
  SED_STR='s#snapshot_prefix: "[a-zA-Z0-9/_.]*"#snapshot_prefix: "'"${1}"'/googlenetv2_snapshot"#g'
  sed -i ''"${SED_STR}"'' ./examples/geoanimal/13parts/${NET}_solver.prototxt.template 
  cp examples/geoanimal/13parts/train_${NET}.sh $1/.
  cp examples/geoanimal/13parts/${NET}_train_val.prototxt.template $1/.
  cp examples/geoanimal/13parts/${NET}_solver.prototxt.template $1/.
  cp examples/geoanimal/13parts/machinefile $1/.
  cp examples/geoanimal/13parts/ps_config_${NET} $1/.
  LOG=$1/output.txt
  OUTDIR=$1
fi

echo '' > ${LOG}

python scripts/duplicate.py examples/geoanimal/13parts/${NET}_train_val.prototxt 13
python scripts/duplicate.py examples/geoanimal/13parts/${NET}_solver.prototxt 13

# Copy files to all the other worker nodes, only do this when there is no
# shared file system among worker nodes
# Note: The first line in the machinefile must be the master node!
readarray -t nodes < examples/geoanimal/13parts/machinefile

for n in "${nodes[@]:1}"
do
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/geoanimal/13parts/machinefile ${n}:$(pwd)/examples/geoanimal/13parts/
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/geoanimal/13parts/ps_config_${NET} ${n}:$(pwd)/examples/geoanimal/13parts/
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/geoanimal/13parts/${NET}_train_val.prototxt.* ${n}:$(pwd)/examples/geoanimal/13parts/
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null examples/geoanimal/13parts/${NET}_solver.prototxt.* ${n}:$(pwd)/examples/geoanimal/13parts/
done


pdsh -R ssh -w ^examples/geoanimal/13parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/geoanimal/13parts/${NET}_solver.prototxt --ps_config=examples/geoanimal/13parts/ps_config_${NET} --machinefile=examples/geoanimal/13parts/machinefile --worker_id=%n --outdir=${OUTDIR}" 2>&1 | tee ${LOG}

