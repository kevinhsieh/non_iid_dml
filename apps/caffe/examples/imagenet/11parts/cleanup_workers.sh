#!/usr/bin/env sh

pdsh -R ssh -w ^examples/imagenet/11parts/machinefile "pkill caffe_geeps"
