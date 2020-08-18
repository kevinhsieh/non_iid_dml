#!/bin/bash

work_dir=$PWD

sed -i 's/enable_gaia=[0-9]\+/enable_gaia=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/num_dc=[0-9]\+/num_dc=5/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/enable_overlay_network=[0-9]\+/enable_overlay_network=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/layers_per_table=[0-9]\+/layers_per_table=5/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/enable_olnw_multiple_routers=[0-9]\+/enable_olnw_multiple_routers=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/slack_table_limit=-\?[0-9]\+/slack_table_limit=-1/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/iters_reach_lower_bound=-\?[0-9]\+/iters_reach_lower_bound=-1/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/slack=[0-9]\+/slack=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/flush_mirror_update_per_iter=-\?[0-9]\+/flush_mirror_update_per_iter=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/mirror_update_threshold=-\?[0-9]\+.[0-9]\+/mirror_update_threshold=0.01/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/local_model_only=[0-9]\+/local_model_only=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/mirror_update_value_threshold=[.0-9]\+/mirror_update_value_threshold=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/lower_update_threshold=[.0-9]\+/lower_update_threshold=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/lower_update_table_limit=[.0-9]\+/lower_update_table_limit=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/model_traveling_freq=[.0-9]\+/model_traveling_freq=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/enable_fedavg=[.0-9]\+/enable_fedavg=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/enable_dgc=[.0-9]\+/enable_dgc=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/gradient_clip_threshold=[.0-9]\+/gradient_clip_threshold=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

#### BSP ####
#sed -i 's#5parts_[a-z_0-9]*/#5parts_shuffled/#g' ./examples/geoanimal/5parts/googlenetv2_train_val.prototxt.template
#./examples/geoanimal/5parts/train_googlenetv2.sh geoanimal_googlenetv2_5parts_bsp_shuffleddata;

#sed -i 's#5parts_[a-z_0-9]*/#5parts_skewed/#g' ./examples/geoanimal/5parts/googlenetv2_train_val.prototxt.template
#./examples/geoanimal/5parts/train_googlenetv2.sh geoanimal_googlenetv2_5parts_bsp_skeweddata;

#### Gaia ####

sed -i 's/enable_gaia=[.0-9]\+/enable_gaia=1/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

for lm in 1; do
    sed -i 's/apply_change_to_local_model=[.0-9]\+/apply_change_to_local_model='"${lm}"'/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

    for th in 0.10 0.20 0.30; do
        sed -i 's/mirror_update_threshold=-\?[0-9]\+.[0-9]\+/mirror_update_threshold='"${th}"'/g' ./examples/geoanimal/5parts/ps_config_googlenetv2;

        sed -i 's#5parts_[a-z_0-9]*/#5parts_skewed/#g' ./examples/geoanimal/5parts/googlenetv2_train_val.prototxt.template
        ./examples/geoanimal/5parts/train_googlenetv2.sh geoanimal_googlenetv2_5parts_gaia_th_${th}_lm_${lm}_skeweddata;

        sed -i 's#5parts_[a-z_0-9]*/#5parts_shuffled/#g' ./examples/geoanimal/5parts/googlenetv2_train_val.prototxt.template
        ./examples/geoanimal/5parts/train_googlenetv2.sh geoanimal_googlenetv2_5parts_gaia_th_${th}_lm_${lm}_shuffleddata;        
    done
done

#### FedAvg ####

sed -i 's/enable_fedavg=[.0-9]\+/enable_fedavg=1/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

sed -i 's/enable_gaia=[.0-9]\+/enable_gaia=0/g' ./examples/geoanimal/5parts/ps_config_googlenetv2

for iter in 100 200 500; do
    sed -i 's/fedavg_local_iter=[.0-9]\+/fedavg_local_iter='"${iter}"'/g' ./examples/geoanimal/5parts/ps_config_googlenetv2;

    sed -i 's#5parts_[a-z_0-9]*/#5parts_skewed/#g' ./examples/geoanimal/5parts/googlenetv2_train_val.prototxt.template
    ./examples/geoanimal/5parts/train_googlenetv2.sh geoanimal_googlenetv2_5parts_fedavg_iter_${iter}_skeweddata;

    sed -i 's#5parts_[a-z_0-9]*/#5parts_shuffled/#g' ./examples/geoanimal/5parts/googlenetv2_train_val.prototxt.template
    ./examples/geoanimal/5parts/train_googlenetv2.sh geoanimal_googlenetv2_5parts_fedavg_iter_${iter}_shuffleddata;
done
