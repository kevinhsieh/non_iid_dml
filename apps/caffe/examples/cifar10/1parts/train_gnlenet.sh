#!/usr/bin/env sh

pdsh -R ssh -p 3022 -w ^examples/cifar10/1parts/machinefile "pkill caffe_geeps"

LOG=output.txt
OUTDIR=

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  SED_STR='s#snapshot_prefix: "[a-zA-Z0-9/_.,]*"#snapshot_prefix: "'"${1}"'/cifar_snapshot"#g'
  sed -i ''"${SED_STR}"'' ./examples/cifar10/1parts/gnlenet_solver.prototxt.template
  cp examples/cifar10/1parts/train_gnlenet.sh $1/.
  cp examples/cifar10/1parts/gnlenet_train_val.prototxt.template $1/.
  cp examples/cifar10/1parts/gnlenet_solver.prototxt.template $1/.
  cp examples/cifar10/1parts/machinefile $1/.
  cp examples/cifar10/1parts/ps_config_gnlenet $1/.
  LOG=$1/output.txt
  OUTDIR=$1
fi

python scripts/duplicate.py examples/cifar10/1parts/gnlenet_train_val.prototxt 1
python scripts/duplicate.py examples/cifar10/1parts/gnlenet_solver.prototxt 1

pdsh -R ssh -p 3022 -w ^examples/cifar10/1parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/1parts/gnlenet_solver.prototxt --ps_config=examples/cifar10/1parts/ps_config_gnlenet --machinefile=examples/cifar10/1parts/machinefile --worker_id=%n --outdir=${OUTDIR}" 2>&1 | tee ${LOG}
