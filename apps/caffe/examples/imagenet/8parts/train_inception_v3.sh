#!/usr/bin/env sh

pdsh -R ssh -w ^examples/imagenet/8parts/machinefile "pkill caffe_geeps"

python scripts/duplicate.py examples/imagenet/8parts/inception_v3_train_val.prototxt 8
python scripts/duplicate.py examples/imagenet/8parts/inception_v3_solver.prototxt 8

LOG=output.txt

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  cp examples/imagenet/8parts/train_inception_v3.sh $1/.
  cp examples/imagenet/8parts/inception_v3_train_val.prototxt.template $1/.
  cp examples/imagenet/8parts/inception_v3_solver.prototxt.template $1/.
  cp examples/imagenet/8parts/machinefile $1/.
  cp examples/imagenet/8parts/ps_config_inception_v3 $1/.
  LOG=$1/output.txt
fi

pdsh -R ssh -w ^examples/imagenet/8parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/imagenet/8parts/inception_v3_solver.prototxt --ps_config=examples/imagenet/8parts/ps_config_inception_v3 --machinefile=examples/imagenet/8parts/machinefile --worker_id=%n" 2>&1 | tee ${LOG}
