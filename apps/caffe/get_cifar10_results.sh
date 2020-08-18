#!/bin/bash

net=${1}
p=${2}
folder=${3}

echo "BSP"

for data in shuffledata skewdata; do
    python get_cifar_result.py ${p} ${folder}/cifar10_${net}_${p}parts_bsp_${data}/output.txt
done

echo "Gaia"

for data in shuffledata skeweddata; do
    for th in 0.02 0.05 0.10 0.20 0.30; do
        python get_cifar_result.py ${p} ${folder}/cifar10_${net}_${p}parts_gaia_th_${th}_lm_1_${data}/output.txt
    done
    echo "---"
done

echo "FedAvg"

for data in shuffledata skeweddata; do
    for iter in 5 10 50 200 1000; do
        python get_cifar_result.py ${p} ${folder}/cifar10_${net}_${p}parts_fedavg_iter_${iter}_${data}/output.txt
    done
    echo "---"
done

echo "DGC"

for data in shuffledata skeweddata; do
    for e in 1000 2000 8000; do
        python get_cifar_result.py ${p} ${folder}/cifar10_${net}_${p}parts_dgc_epoch_${e}_lm_0_dmm_0_${data}/output.txt
    done
    echo "---"
done

