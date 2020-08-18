#!/usr/bin/env sh

pdsh -R ssh -w ^examples/imagenet/8parts/machinefile "pkill caffe_geeps"
