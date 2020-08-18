#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/1parts/create_cifar10.sh 1
pdsh -R ssh -w ^examples/cifar10/1parts/machinefile "cd $(pwd) && ./examples/cifar10/1parts/create_cifar10.sh.%n"
