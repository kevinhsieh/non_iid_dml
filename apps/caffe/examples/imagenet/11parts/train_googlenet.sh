#!/usr/bin/env sh

pdsh -R ssh -w ^examples/imagenet/11parts/machinefile "pkill caffe_geeps"

python scripts/duplicate.py examples/imagenet/11parts/googlenet_train_val.prototxt 11
python scripts/duplicate.py examples/imagenet/11parts/googlenet_solver.prototxt 11

LOG=output.txt

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  cp examples/imagenet/11parts/train_googlenet.sh $1/.
  cp examples/imagenet/11parts/googlenet_train_val.prototxt.template $1/.
  cp examples/imagenet/11parts/googlenet_solver.prototxt.template $1/.
  cp examples/imagenet/11parts/machinefile $1/.
  cp examples/imagenet/11parts/ps_config_googlenet $1/.
  LOG=$1/output.txt
fi

pdsh -R ssh -w ^examples/imagenet/11parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/imagenet/11parts/googlenet_solver.prototxt --ps_config=examples/imagenet/11parts/ps_config_googlenet --machinefile=examples/imagenet/11parts/machinefile --worker_id=%n" 2>&1 | tee ${LOG}
