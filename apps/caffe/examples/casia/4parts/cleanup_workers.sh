#!/usr/bin/env sh

pdsh -R ssh -w ^examples/casia/4parts/machinefile "pkill caffe_geeps"
