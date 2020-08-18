# The Non-IID Data Quagmire of Decentralized Machine Learning 

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This repo is the source code for our paper: [The Non-IID Data Quagmire of Decentralized Machine Learning (ICML'20)](https://icml.cc/virtual/2020/poster/6306). This repo also contains our implementation of [Gaia (NSDI'17)](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/hsieh).

This source code is built on a caffe-based parameter server system, [GeePS](https://github.com/cuihenggang/geeps).

The following steps assume compatible CUDA and CuDNN are installed. The code is tested on Ubuntu 16.04 with CUDA 10.2 and CuDNN 7.6.5.

If you use docker, you can start with `nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04`.

## Build the application

As the following steps need to install dependencies, we recommend the user executes these steps only in a clean machine or docker container.

First, clone the project repo and switch into the root directory:

```
git clone https://github.com/kevinhsieh/non_iid_dml.git
cd non_iid_dml
```

If you use the Ubuntu 16.04 system, you can run the following commands to install the dependencies:

```
./scripts/install-geeps-deps-ubuntu16.sh
./scripts/install-caffe-deps-ubuntu16.sh
```

Also, please make sure your CUDA library is installed in `/usr/local/cuda`.

Note that all the nodes for the experiments need to install the aforementioned dependencies.

After installing the dependencies, you can build the application by simply running these commands:

```
cd apps/caffe
./make_all.sh
```

You can optionally create an VM image after above steps if you run this code on a public cloud. This will make sure all the nodes have the same environment and built binary to run the experiments. Example for Azure can be found [here](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/capture-image).


## Run CIFAR-10 on two machines in the IID and Non-IID settings


All commands in this section are executed from the `apps/caffe` directory:

```
cd apps/caffe
```

You will first need to prepare a machine file as `examples/cifar10/2parts/machinefile`, with each line being the host name (or IP) of one machine. Since we use two machines in this example, this machine file should have two lines, such as:

```
h0
h1
```

We will use `pdsh` to launch commands on those machines with the `ssh` protocol, so please make sure that you can `ssh` to those machines *without* password (e.g., using private keys). You can also optionally use this command to eliminate errors from SSH:

```
export PDSH_SSH_ARGS_APPEND="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
```

When you have your machine file in ready, you can run the following command to download and prepare the CIFAR-10 dataset:

```
./data/cifar10/get_cifar10.sh
./examples/cifar10/2parts/create_cifar10_pdsh.sh
```

Our script will partition the dataset into two sets of partitions. The partitions for the IID setting is in `./examples/cifar10/2parts/shuffled`, and the ones for the Non-IID setting is in `./examples/cifar10/2parts/skewed`.

You need to copy these newly created data partitions to the other node. For example:

```
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r ./examples/cifar10/2parts/shuffled h1:$(pwd)/examples/cifar10/2parts/
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r ./examples/cifar10/2parts/skewed h1:$(pwd)/examples/cifar10/2parts/
```

Finally, you should set up an output folder that can be accessed by the nodes. For example:

```
export OUTPUT_DATA_PATH="$(pwd)/outputs"
```

Once the data partitions are ready, you can execute the following command to run different decentralized learning algorithms and DNNs in the IID and Non-IID settings. For example, this command runs BSP, DeepGradientCompression, Gaia, and FederatedAveraging for GN-LeNet on 2 partitions: 

```
./run_cifar10_exps.sh gnlenet 2
```

The script `run_cifar10_exps.sh` contains all the details of configurations. You can change this script to run various hyper-parameters and configurations.

Once the experiment is done, you can get the validation accuracy results by running

```
python get_cifar_result_2parts_curve.py <output folder>/output.txt
```

In the paper, we use 5 partitions for CIFAR-10. To do so, you can repeat above steps with 5 machines and change the number of partitions from 2 to 5.

The `example` folder also contains the example codes for other datasets and applications such as ImageNet (`imagenet`), Face Recognition (`casia`), and our [Flickr-Mammal dataset](https://doi.org/10.5281/zenodo.3676081) (`geoanimal`).


## Reference Papers

If you use our code in your work, we would appreciate a reference to the following papers

Kevin Hsieh, Amar Phanishayee, Onur Mutlu, and Phillip B Gibbons. [The Non-IID Data Quagmire of Decentralized Machine Learning](https://proceedings.icml.cc/static/paper_files/icml/2020/3152-Paper.pdf). Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Kevin Hsieh, Aaron Harlap, Nandita Vijaykumar, Dimitris Konomis, Gregory R. Ganger, Phillip B. Gibbons and Onur Mutlu. [Gaia: Geo-Distributed Machine Learning Approaching LAN Speeds](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-hsieh.pdf). Proceedings of the 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI), 2017.