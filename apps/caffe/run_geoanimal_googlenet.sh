#!/bin/bash

work_dir=$PWD
p=${1}

if [ -z "${OUTPUT_DATA_PATH}" ]
then
    echo "OUTPUT_DATA_PATH is not set, terminating.."
    exit 1
fi

sed -i 's/enable_gaia=[0-9]\+/enable_gaia=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/num_dc=[0-9]\+/num_dc='"${p}"'/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/enable_overlay_network=[0-9]\+/enable_overlay_network=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/layers_per_table=[0-9]\+/layers_per_table=5/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/enable_olnw_multiple_routers=[0-9]\+/enable_olnw_multiple_routers=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/slack_table_limit=-\?[0-9]\+/slack_table_limit=-1/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/iters_reach_lower_bound=-\?[0-9]\+/iters_reach_lower_bound=-1/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/slack=[0-9]\+/slack=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/flush_mirror_update_per_iter=-\?[0-9]\+/flush_mirror_update_per_iter=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/mirror_update_threshold=-\?[0-9]\+.[0-9]\+/mirror_update_threshold=0.01/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/local_model_only=[0-9]\+/local_model_only=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/mirror_update_value_threshold=[.0-9]\+/mirror_update_value_threshold=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/lower_update_threshold=[.0-9]\+/lower_update_threshold=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/lower_update_table_limit=[.0-9]\+/lower_update_table_limit=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/model_traveling_freq=[.0-9]\+/model_traveling_freq=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/enable_fedavg=[.0-9]\+/enable_fedavg=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/enable_dgc=[.0-9]\+/enable_dgc=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/gradient_clip_threshold=[.0-9]\+/gradient_clip_threshold=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

#### BSP ####
sed -i 's#'"${p}"'parts_[a-z_0-9]*/#'"${p}"'parts_shuffled/#g' ./examples/geoanimal/${p}parts/googlenetv2_train_val.prototxt.template
./examples/geoanimal/${p}parts/train_googlenetv2.sh ${OUTPUT_DATA_PATH}/geoanimal_googlenetv2_${p}parts_bsp_shuffleddata;

sed -i 's#'"${p}"'parts_[a-z_0-9]*/#'"${p}"'parts_skewed/#g' ./examples/geoanimal/${p}parts/googlenetv2_train_val.prototxt.template
./examples/geoanimal/${p}parts/train_googlenetv2.sh ${OUTPUT_DATA_PATH}/geoanimal_googlenetv2_${p}parts_bsp_skeweddata;

#### Gaia ####

sed -i 's/enable_gaia=[.0-9]\+/enable_gaia=1/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

for lm in 1; do
    sed -i 's/apply_change_to_local_model=[.0-9]\+/apply_change_to_local_model='"${lm}"'/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

    for th in 0.05 0.10 0.20; do
        sed -i 's/mirror_update_threshold=-\?[0-9]\+.[0-9]\+/mirror_update_threshold='"${th}"'/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2;

        sed -i 's#'"${p}"'parts_[a-z_0-9]*/#'"${p}"'parts_shuffled/#g' ./examples/geoanimal/${p}parts/googlenetv2_train_val.prototxt.template
        ./examples/geoanimal/${p}parts/train_googlenetv2.sh ${OUTPUT_DATA_PATH}/geoanimal_googlenetv2_${p}parts_gaia_th_${th}_lm_${lm}_shuffleddata;

        sed -i 's#'"${p}"'parts_[a-z_0-9]*/#'"${p}"'parts_skewed/#g' ./examples/geoanimal/${p}parts/googlenetv2_train_val.prototxt.template
        ./examples/geoanimal/${p}parts/train_googlenetv2.sh ${OUTPUT_DATA_PATH}/geoanimal_googlenetv2_${p}parts_gaia_th_${th}_lm_${lm}_skeweddata;
    done
done


#### FedAvg ####

sed -i 's/enable_fedavg=[.0-9]\+/enable_fedavg=1/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

sed -i 's/enable_gaia=[.0-9]\+/enable_gaia=0/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2

for iter in 50 100 200; do
    sed -i 's/fedavg_local_iter=[.0-9]\+/fedavg_local_iter='"${iter}"'/g' ./examples/geoanimal/${p}parts/ps_config_googlenetv2;

    sed -i 's#'"${p}"'parts_[a-z_0-9]*/#'"${p}"'parts_shuffled/#g' ./examples/geoanimal/${p}parts/googlenetv2_train_val.prototxt.template
    ./examples/geoanimal/${p}parts/train_googlenetv2.sh ${OUTPUT_DATA_PATH}/geoanimal_googlenetv2_${p}parts_fedavg_iter_${iter}_shuffleddata;

    sed -i 's#'"${p}"'parts_[a-z_0-9]*/#'"${p}"'parts_skewed/#g' ./examples/geoanimal/${p}parts/googlenetv2_train_val.prototxt.template
    ./examples/geoanimal/${p}parts/train_googlenetv2.sh ${OUTPUT_DATA_PATH}/geoanimal_googlenetv2_${p}parts_fedavg_iter_${iter}_skeweddata;
done


