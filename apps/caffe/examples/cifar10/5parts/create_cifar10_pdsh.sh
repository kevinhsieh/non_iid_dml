#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/5parts/create_cifar10.sh 5
pdsh -R ssh -w ^examples/cifar10/5parts/machinefile "cd $(pwd) && ./examples/cifar10/5parts/create_cifar10.sh.%n"
