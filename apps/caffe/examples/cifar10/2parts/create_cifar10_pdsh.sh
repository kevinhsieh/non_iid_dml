#!/usr/bin/env bash

python scripts/duplicate.py examples/cifar10/2parts/create_cifar10.sh 2
for n in 0 1;
do
    ./examples/cifar10/2parts/create_cifar10.sh.$n
done
#pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "cd $(pwd) && ./examples/cifar10/2parts/create_cifar10.sh.%n"
