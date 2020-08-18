#!/usr/bin/env sh

pdsh -R ssh -w ^examples/geoanimal/5parts/machinefile "pkill caffe_geeps"
