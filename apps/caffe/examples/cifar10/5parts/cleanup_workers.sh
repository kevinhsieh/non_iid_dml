#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/5parts/machinefile "pkill caffe_geeps"
