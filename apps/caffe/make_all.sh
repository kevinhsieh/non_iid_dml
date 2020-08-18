#!/bin/bash

cd ../../
scons -j4
cd apps/caffe
rm build/tools/*.bin
make -j4

