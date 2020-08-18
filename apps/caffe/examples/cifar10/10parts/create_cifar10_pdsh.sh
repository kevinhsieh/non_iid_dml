#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/10parts/create_cifar10.sh 10

for n in 0 1 2 3 4 5 6 7 8 9
do
    ./examples/cifar10/10parts/create_cifar10.sh.${n}
done

#pdsh -R ssh -w ^examples/cifar10/10parts/machinefile "cd $(pwd) && ./examples/cifar10/10parts/create_cifar10.sh.%n"
