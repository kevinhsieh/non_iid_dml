#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "pkill caffe_geeps"
