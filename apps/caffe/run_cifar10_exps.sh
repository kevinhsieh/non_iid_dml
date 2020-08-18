#!/bin/bash

work_dir=$PWD
net=${1}
p=${2}

if [ -z "${OUTPUT_DATA_PATH}" ]
then
    echo "OUTPUT_DATA_PATH is not set, terminating.."
    exit 1
fi

sed -i 's/enable_gaia=[0-9]\+/enable_gaia=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/num_dc=[0-9]\+/num_dc='"${p}"'/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/enable_overlay_network=[0-9]\+/enable_overlay_network=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/layers_per_table=[0-9]\+/layers_per_table=5/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/enable_olnw_multiple_routers=[0-9]\+/enable_olnw_multiple_routers=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/slack_table_limit=-\?[0-9]\+/slack_table_limit=-1/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/iters_reach_lower_bound=-\?[0-9]\+/iters_reach_lower_bound=-1/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/slack=[0-9]\+/slack=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/flush_mirror_update_per_iter=-\?[0-9]\+/flush_mirror_update_per_iter=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/mirror_update_threshold=-\?[0-9]\+.[0-9]\+/mirror_update_threshold=0.01/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/local_model_only=[0-9]\+/local_model_only=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/mirror_update_value_threshold=[.0-9]\+/mirror_update_value_threshold=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/lower_update_threshold=[.0-9]\+/lower_update_threshold=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/lower_update_table_limit=[.0-9]\+/lower_update_table_limit=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/model_traveling_freq=[.0-9]\+/model_traveling_freq=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/enable_fedavg=[.0-9]\+/enable_fedavg=0/g' ./examples/cifar10/${p}parts/ps_config_${net}


#### BSP ####

sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/shuffled/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_bsp_shuffledata;

sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/skewed/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_bsp_skewdata;

#### DGC ####

sed -i 's/enable_dgc=[.0-9]\+/enable_dgc=1/g' ./examples/cifar10/${p}parts/ps_config_${net}

for epoch in 1000; do
    sed -i 's/dgc_epoch_size=[.0-9]\+/dgc_epoch_size='"${epoch}"'/g' ./examples/cifar10/${p}parts/ps_config_${net};
    for lm in 0; do
        sed -i 's/apply_change_to_local_model=[.0-9]\+/apply_change_to_local_model='"${lm}"'/g' ./examples/cifar10/${p}parts/ps_config_${net};
        for dmm in 0; do
            sed -i 's/disable_dgc_momemtum_mask=[.0-9]\+/disable_dgc_momemtum_mask='"${dmm}"'/g' ./examples/cifar10/${p}parts/ps_config_${net};

            sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/shuffled/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
            ./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_dgc_epoch_${epoch}_lm_${lm}_dmm_${dmm}_shuffledata;

            sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/skewed/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
            ./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_dgc_epoch_${epoch}_lm_${lm}_dmm_${dmm}_skeweddata;
        done
    done
done

#### Gaia ####

sed -i 's/enable_gaia=[.0-9]\+/enable_gaia=1/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/enable_dgc=[.0-9]\+/enable_dgc=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

for lm in 1; do
    sed -i 's/apply_change_to_local_model=[.0-9]\+/apply_change_to_local_model='"${lm}"'/g' ./examples/cifar10/${p}parts/ps_config_${net}

    for th in 0.20; do
        sed -i 's/mirror_update_threshold=-\?[0-9]\+.[0-9]\+/mirror_update_threshold='"${th}"'/g' ./examples/cifar10/${p}parts/ps_config_${net};

        sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/shuffled/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
        ./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_gaia_th_${th}_lm_${lm}_shuffledata;

        sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/skewed/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
        ./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_gaia_th_${th}_lm_${lm}_skeweddata;
    done
done

#### FedAvg ####

sed -i 's/enable_fedavg=[.0-9]\+/enable_fedavg=1/g' ./examples/cifar10/${p}parts/ps_config_${net}

sed -i 's/enable_gaia=[.0-9]\+/enable_gaia=0/g' ./examples/cifar10/${p}parts/ps_config_${net}

for iter in 50; do
    sed -i 's/fedavg_local_iter=[.0-9]\+/fedavg_local_iter='"${iter}"'/g' ./examples/cifar10/${p}parts/ps_config_${net};

    sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/shuffled/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
    ./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_fedavg_iter_${iter}_shuffledata;

    sed -i 's#'"${p}"'parts/[a-z_0-9]*/#'"${p}"'parts/skewed/#g' ./examples/cifar10/${p}parts/${net}_train_val.prototxt.template
    ./examples/cifar10/${p}parts/train_net.sh ${net} ${OUTPUT_DATA_PATH}/cifar10_${net}_${p}parts_fedavg_iter_${iter}_skeweddata;
done

