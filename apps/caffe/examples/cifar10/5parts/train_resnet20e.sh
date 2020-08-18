#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/5parts/machinefile "pkill caffe_geeps"

LOG=output.txt
OUTDIR=

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  SED_STR='s#snapshot_prefix: "[a-zA-Z0-9/_.,]*"#snapshot_prefix: "'"${1}"'/cifar_snapshot"#g'
  sed -i ''"${SED_STR}"'' ./examples/cifar10/5parts/resnet20e_solver.prototxt.template
  cp examples/cifar10/5parts/train_resnet20e.sh $1/.
  cp examples/cifar10/5parts/resnet20e_train_val.prototxt.template $1/.
  cp examples/cifar10/5parts/resnet20e_solver.prototxt.template $1/.
  cp examples/cifar10/5parts/machinefile $1/.
  cp examples/cifar10/5parts/ps_config_resnet20e $1/.
  LOG=$1/output.txt
  OUTDIR=$1
fi

python scripts/duplicate.py examples/cifar10/5parts/resnet20e_train_val.prototxt 5
python scripts/duplicate.py examples/cifar10/5parts/resnet20e_solver.prototxt 5

pdsh -R ssh -w ^examples/cifar10/5parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/5parts/resnet20e_solver.prototxt --ps_config=examples/cifar10/5parts/ps_config_resnet20e --machinefile=examples/cifar10/5parts/machinefile --worker_id=%n --outdir=${OUTDIR}" 2>&1 | tee ${LOG}
