#!/usr/bin/env sh

pdsh -R ssh -w ^examples/geoanimal/13parts/machinefile "pkill caffe_geeps"
