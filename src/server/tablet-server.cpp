/*
 * Copyright (c) 2016, Anonymous Institution.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/* GeePS tablet server */

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <utility>
#include <string>
#include <vector>
#include <functional>
#include <queue>

#include "common/internal-config.hpp"
#include "server-encoder-decoder.hpp"
#include "mirror-server-encoder-decoder.hpp"
#include "tablet-server.hpp"

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::make_pair;
using boost::format;
using boost::lexical_cast;
using boost::shared_ptr;
using boost::make_shared;

#define DEBUG_MODEL_TRAVEL (0)

TabletStorage::TabletStorage(
	uint channel_id, uint num_channels, uint process_id, uint num_processes,
	shared_ptr<ServerClientEncode> communicator,
	shared_ptr<MirrorClientEncode> mc_communicator,
	cudaStream_t cuda_stream, cublasHandle_t cublas_handle,
	const GeePsConfig& config) :
	channel_id(channel_id), num_channels(num_channels),
	process_id(process_id), num_processes(num_processes),
	num_clients(num_processes),
	communicator(communicator),
	mc_communicator(mc_communicator),
	cuda_stream(cuda_stream), cublas_handle(cublas_handle),
	config(config) 
{
	begin_time = tbb::tick_count::now();

  /* Initialize data tables */
  data_tables.resize(config.num_tables);
  for (uint table_id = 0; table_id < data_tables.size(); table_id++) {
    DataTable& data_table = data_tables[table_id];
    data_table.vec_clock.resize(num_processes);
		data_table.mirror_clock.resize(config.num_dc);
		data_table.dcg_clock.resize(config.num_dcg);
    for (uint client_id = 0; client_id < num_processes; client_id++) {
      data_table.vec_clock[client_id] = INITIAL_DATA_AGE;
    }
		for (uint dc_id = 0; dc_id < config.num_dc; dc_id++) {
      data_table.mirror_clock[dc_id] = INITIAL_DATA_AGE;
    }
		for (uint dcg_id = 0; dcg_id < config.num_dcg; dcg_id++) {
      data_table.dcg_clock[dcg_id] = INITIAL_DATA_AGE;
    }
    data_table.travel_leave_clock = INITIAL_DATA_AGE;
    data_table.travel_arrive_clock = INITIAL_DATA_AGE;
    data_table.travel_apply_clock = INITIAL_DATA_AGE;
    data_table.global_clock = INITIAL_DATA_AGE;
		data_table.inter_dcg_clock = INITIAL_DATA_AGE;
		data_table.local_clock = INITIAL_DATA_AGE;		
    data_table.row_count = 0;
  }
	tablet_mutex = make_shared<boost::mutex>();
	last_mirror_send_size = 0;
	iter_mirror_send_size = 0;
	update_threshold_adj = 0;
	top_layer_send_clock = INITIAL_DATA_AGE;
	wan_bandwidth_limit_byte_per_sec = 
		config.wan_bandwidth_limit * 1000000 / num_channels / 8;

	if (config.aggr_mirror_update_table_group > 0) {
		switch(config.aggr_mirror_update_table_group)
		{
		case 1:
			aggr_table_group.insert(4);
			aggr_table_group.insert(8);
			aggr_table_group.insert(11);
			aggr_table_group.insert(12);
			break;
		case 2:
			aggr_table_group.insert(4);
			aggr_table_group.insert(8);
			aggr_table_group.insert(11);
			break;
    case 3:
			aggr_table_group.insert(4);
			aggr_table_group.insert(8);
			break;
    case 4:
			aggr_table_group.insert(4);
			break;
		case 5:
			aggr_table_group.insert(8);
			break;
		}
	}

	is_dcg_router = false;
	if (config.enable_overlay_network) {
		is_dcg_router = config.dcg_router_list[get_dc_id(process_id)];
	}
}


void TabletStorage::update_row_batch(
	uint client_id, iter_t clock, uint table_id,
	RowKey *row_keys, RowOpVal *updates, uint batch_size) {

	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

  server_stats.nr_update += batch_size;
  if (client_id == process_id) {
    server_stats.nr_local_update += batch_size;
  }

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;
  
  CHECK_LT(client_id, data_table.vec_clock.size());
  iter_t cur_clock = data_table.vec_clock[client_id];
  if (cur_clock != INITIAL_DATA_AGE && clock != cur_clock + 1) {
    cerr << "WARNING CS clocks out of sync,"
         << " client = " << client_id
         << " clock = " << clock
         << " cur_clock = " << cur_clock
         << endl;
    CHECK(0);
  }

  if (batch_size == 0) {
    return;
  }

  if (data_store.size() == 0) {
    data_store.init(batch_size, DataStorage::CPU);
    data_store.zerofy_data_cpu();
    data_table.local_update.init(batch_size, DataStorage::CPU);
    data_table.local_update.zerofy_data_cpu();
    data_table.update_history.init(batch_size, DataStorage::CPU);
    data_table.update_history.zerofy_data_cpu();
    data_table.row_count = batch_size;
    memcpy(data_store.row_keys.data(), row_keys,
					 batch_size * sizeof(RowKey));
		data_table.accu_update.init(batch_size, DataStorage::CPU);
		data_table.accu_update.zerofy_data_cpu();
		data_table.intra_dcg_accu_update.init(batch_size, DataStorage::CPU);
		data_table.intra_dcg_accu_update.zerofy_data_cpu();
		data_table.inter_dcg_accu_update.init(batch_size, DataStorage::CPU);
		data_table.inter_dcg_accu_update.zerofy_data_cpu();
    data_table.travel_model_store.init(batch_size, DataStorage::CPU);
		data_table.travel_model_store.zerofy_data_cpu();
  }

#if (ENABLE_UPDATE_DIST)
	for (int c = 0; c < UPDATE_DIST_COL_NUM; c++) 
	{
		if (data_table.shadow_accu_update[c].size() == 0) 
		{
			data_table.shadow_accu_update[c].init(batch_size, DataStorage::CPU);
			data_table.shadow_accu_update[c].zerofy_data_cpu();
		}
	}
#endif

  CHECK_EQ(data_store.size(), batch_size);
  apply_updates(table_id, updates, batch_size, clock);
}

void TabletStorage::apply_updates(
	uint table_id, RowOpVal *update_rows, size_t batch_size, iter_t clock) 
{

	/* This function is called ONLY when the mutex is held */

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
#if (HISTORY_IN_SERVER)
  DataStorage& data_store = data_table.local_update;
#else
  DataStorage& data_store = data_table.store;
#endif
  
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(data_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;  
  
  cpu_add(num_vals,
					master_data,
					update,
					master_data);

#if 0
  ParamOffset& param_offset_list = data_table.param_offset_list;
  for (size_t param_idx = 0; param_idx < param_offset_list.size(); param_idx++) {
    ParamTableRowOffset& param_offset = param_offset_list[param_idx];
    size_t offset = param_offset.start_row_id * ROW_DATA_SIZE 
        + param_offset.val_offset;
    if (60 == param_offset.global_param_id) {
      LOG(INFO) << "Param: " << param_offset.global_param_id
                << ", clock: " << clock
                << ", recv update: " << *(update + offset)
                << ", local update: " << *(master_data + offset);
    }
  }
#endif

#if !(HISTORY_IN_SERVER)
  /* We handle Gaia stuff here only when the history is kept 
     on the clients */
	if (config.enable_gaia && clock != 0) {
    apply_gaia_updates(update, num_vals, clock);
	}
#endif
}

void TabletStorage::apply_gaia_updates(uint table_id, val_t *update, size_t num_vals, 
                                       iter_t clock)
{
  CHECK_LT(table_id, data_tables.size());

  DataTable& data_table = data_tables[table_id];

  // Add to the accumalated updates
  val_t *accu_update = reinterpret_cast<val_t *>(
    data_table.accu_update.data());
  cpu_add(num_vals,
          accu_update,
          update,
          accu_update);

  server_stats.total_update[clock/UPDATE_DIST_ITER_INTERVAL]+=num_vals;

#if (ENABLE_UPDATE_DIST)
  for (int c = 0; c < UPDATE_DIST_COL_NUM; c++) 
  {
    // Add to the accumalated updates
    val_t *shadow_update = reinterpret_cast<val_t *>(
      data_table.shadow_accu_update[c].data());
    cpu_add(num_vals,
            shadow_update,
            update,
            shadow_update);
  }
#endif

  if (!config.merge_local_update) {
    send_mirror_updates(table_id, clock);
  }
}

void TabletStorage::apply_dgc_updates(uint table_id, val_t *update, size_t num_vals, 
                                       iter_t clock)
{
  CHECK_LT(table_id, data_tables.size());

  DataTable& data_table = data_tables[table_id];

  // Add to the accumalated updates
  val_t *accu_update = reinterpret_cast<val_t *>(
    data_table.accu_update.data());
  cpu_add(num_vals,
          accu_update,
          update,
          accu_update);
}

void TabletStorage::apply_fed_avg_updates(uint table_id, val_t *update, size_t num_vals, 
                                         iter_t clock)
{
#if (ENABLE_UPDATE_STAT)
  CHECK_LT(table_id, data_tables.size());

  DataTable& data_table = data_tables[table_id];

  // Add to the accumalated updates
  val_t *accu_update = reinterpret_cast<val_t *>(
    data_table.accu_update.data());
  cpu_add(num_vals,
          accu_update,
          update,
          accu_update);
#endif
}


void TabletStorage::apply_local_updates_to_store(uint table_id, iter_t clock)
{

#if !(HISTORY_IN_SERVER)
  return;
#endif

  /* This function is called ONLY when the mutex is held */
  
  CHECK_LT(table_id, data_tables.size());

  DataTable& data_table = data_tables[table_id];
  val_t *master_data = reinterpret_cast<val_t *>(data_table.store.data());
  val_t *local_update_data = reinterpret_cast<val_t *>(data_table.local_update.data());
  val_t *update_history_data = reinterpret_cast<val_t *>(data_table.update_history.data());
  ParamOffset& param_offset_list = data_table.param_offset_list;
  size_t num_vals = data_table.store.size() * ROW_DATA_SIZE;

  if (num_vals == 0) {
    return;
  }

  if (clock == 0) {
    /* Special case for the first iteration (init values) */

    /* Add local updates to data store */
    cpu_add(num_vals, master_data, local_update_data, master_data);
    /* Clear local update */
    data_table.local_update.zerofy_data_cpu();
    
  } else {

    for (size_t param_idx = 0; param_idx < param_offset_list.size(); param_idx++) {
      ParamTableRowOffset& param_offset = param_offset_list[param_idx];
      float lr = param_learning_rates[param_offset.global_param_id];
      float momentum = param_momentums[param_offset.global_param_id];
      size_t offset = param_offset.start_row_id * ROW_DATA_SIZE 
        + param_offset.val_offset;

#if 0
      if (0 == process_id) {
        LOG(INFO) << "Channel " << channel_id
                  << ", Param " << param_offset.global_param_id 
                  << ", LR = " << lr
                  << ", Offset = " << offset
                  << ", Table = " << table_id
                  << ", Clock = " << data_table.global_clock
                  << ", Val count = " << param_offset.val_count;
      }
#endif
      CHECK_LE(offset + param_offset.val_count, num_vals);

      cpu_axpby(param_offset.val_count,                 
                lr, local_update_data + offset,
                momentum, update_history_data + offset);

#if 0
      if (60 == param_offset.global_param_id) {
        LOG(INFO) << "Param: " << param_offset.global_param_id
                  << ", LR: " << lr
                  << ", local update: " << *(local_update_data + offset)
                  << ", M: " << momentum
                  << ", final: " << *(update_history_data + offset);
      }
#endif

    }

    /* Gradient clipping */    
    if (config.gradient_clip_threshold > 0)
    {
      server_stats.total_grad_clip_check++;

      /* Calculate the L2 norm */
      val_t l2_norm = cpu_dot(num_vals, update_history_data, update_history_data);
      l2_norm = pow(l2_norm, 0.5);
      
      if (l2_norm > (val_t)config.gradient_clip_threshold) 
      {
        cpu_scal(num_vals, 
                 (val_t)config.gradient_clip_threshold / l2_norm,
                 update_history_data);
        server_stats.total_grad_clip_apply++;
      }
      
      if (l2_norm > server_stats.max_grad_l2_norm) 
      {
        server_stats.max_grad_l2_norm = l2_norm;
      }
    }

    /* Add updates to data store */
    if ((!config.enable_dgc && !config.enable_gaia) || 
        config.apply_change_to_local_model) 
    {
      cpu_add(num_vals, master_data, update_history_data, master_data);
    }

    /* Handle Gaia case */ 
    if (config.enable_gaia) {
      apply_gaia_updates(table_id, update_history_data, num_vals, clock);
    }

    /* Handle DGC case */
    if (config.enable_dgc) {
      apply_dgc_updates(table_id, update_history_data, num_vals, clock);
    }

    /* Handle FedAvg case */
    if (config.enable_fedavg) {
      apply_fed_avg_updates(table_id, update_history_data, num_vals, clock);
    }
    
    /* Clear local update */
    data_table.local_update.zerofy_data_cpu();
  }
}

void TabletStorage::apply_fed_avg_weights(uint table_id)
{
  /* This function is called ONLY when the mutex is held */
  CHECK_EQ(config.enable_fedavg, 1);

	CHECK_LT(table_id, data_tables.size());

  DataTable& data_table = data_tables[table_id];
  val_t *master_data = reinterpret_cast<val_t *>(data_table.store.data());
  val_t *other_dc_data = reinterpret_cast<val_t *>(data_table.travel_model_store.data());
  size_t num_vals = data_table.store.size() * ROW_DATA_SIZE;

  /* Average the weights based on the number of DCs */
  cpu_axpby(num_vals, (val_t)1/config.num_dc, other_dc_data,
            (val_t)1/config.num_dc, master_data);

  /* Clear data from other DCs */
  data_table.travel_model_store.zerofy_data_cpu();  

  update_fed_avg_stats(table_id, data_table.global_clock);
}

void TabletStorage::update_fed_avg_stats(uint table_id, iter_t clock)
{
  /* This function is called ONLY when the mutex is held */

#if (ENABLE_UPDATE_STAT)
  
  CHECK_EQ(config.enable_fedavg, 1);

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;

  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
  size_t num_vals = data_store.size() * ROW_DATA_SIZE;
	val_t *accu_update = reinterpret_cast<val_t *>(
    data_table.accu_update.data());

  for (size_t v = 0; v < num_vals; v++) {
    if (accu_update[v] == 0)
      continue;
    double update_ratio;
    if (master_data[v] == 0) {
      update_ratio = 1.0;
    } else {
      update_ratio = std::abs(accu_update[v] / (double)master_data[v]);
    }

    if (!isnan(update_ratio)) {  //NaN check
      server_stats.update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
      server_stats.update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
    }
    if (!isnan(accu_update[v])) { //NaN check
      server_stats.update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
      server_stats.update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
    }
  }

  data_table.accu_update.zerofy_data_cpu();

#endif
}

bool TabletStorage::send_mirror_updates(uint table_id, iter_t clock) 
{

	/* This function is called ONLY when the mutex is held */

	CHECK_EQ(config.enable_gaia, 1);

	CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;
  
  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
  size_t num_vals = data_store.size() * ROW_DATA_SIZE;
	val_t *accu_update = reinterpret_cast<val_t *>(
			data_table.accu_update.data());

	/* Send mirror updates if there are any significant ones */
	vector<col_idx_t> *col_indexes = new vector<col_idx_t>();
	vector<val_t> *col_updates = new vector<val_t>();

#if (ENABLE_UPDATE_DIST)
	calc_update_dist(table_id, clock);
#endif

	float update_threshold = config.mirror_update_threshold;
  float thres_adj_factor = 1.0;

  /* Apply mirror update schedule */
  if (config.gaia_threshold_sched.size() > 0 && clock != INITIAL_DATA_AGE)
  {
    int sched_idx = std::min(clock / config.gaia_threshold_iter, 
                             (iter_t)(config.gaia_threshold_sched.size() - 1));
    update_threshold = config.gaia_threshold_sched[sched_idx];
    if (0 == clock % config.gaia_threshold_iter) 
    {
      LOG(INFO) << "Change Gaia threshold to " << update_threshold;
    }
  }

	if (config.enable_threshold_by_bw && clock > 0)
	{
		if (0 == table_id)
		{
			/* Determine bandwidth consumption for this iteration */
			double time_for_this_iter = 
				(tbb::tick_count::now() - top_layer_mirror_send_time).seconds();
			double bw_usage = (double)iter_mirror_send_size / time_for_this_iter;
			/* We assume lower bound is larger than default */
			double adj_max = config.mirror_update_lower_bound - config.mirror_update_threshold;
			double adj_min = 0;

			if (config.iters_reach_lower_bound > 0 && clock > 0 &&
					config.iters_reach_lower_bound > clock)
			{
				/* Use linear interpolation to reach the lower bound */
				adj_max = (config.mirror_update_lower_bound - config.mirror_update_threshold) * 
					(float)clock / (float)config.iters_reach_lower_bound;
			}

			/* Adjust the threshold based on bandwidth consumption */
			if (bw_usage > wan_bandwidth_limit_byte_per_sec * 1.1)
			{
				update_threshold_adj += 0.005;
			}
			else if (bw_usage < wan_bandwidth_limit_byte_per_sec * 0.9)
			{
				update_threshold_adj -= 0.005;
			}

			/* Enforce the range of threshold adjustment */
			if (update_threshold_adj > adj_max)
			{
				update_threshold_adj = adj_max;
			}

			if (update_threshold_adj < adj_min)
			{
				update_threshold_adj = adj_min;
			}

			server_stats.total_update_threshold_adj[clock/UPDATE_DIST_ITER_INTERVAL] += 
				update_threshold_adj;

#if 0 /* Debug...*/
			if (0 == channel_id && 0 == process_id)
			{
				LOG(INFO) << "Adj update threshold for iter " << clock << " is " 
									<< update_threshold_adj << ", bw usage: " << bw_usage
									<< ", send_size: " << iter_mirror_send_size
									<< ", time: " << time_for_this_iter;
			}
#endif
		}

		update_threshold += update_threshold_adj;
		
	}
	else if (config.aggr_mirror_update_table_group > 0 &&
			aggr_table_group.find(table_id) != aggr_table_group.end())
	{
		update_threshold = config.aggr_mirror_update_threshold;
	}
	else if (config.iters_reach_lower_bound > 0  && clock > 0)
	{
		if (config.iters_reach_lower_bound > clock) 
		{
			/* Use linear interpolation to reach the lower bound */
			update_threshold = config.mirror_update_threshold +
				(config.mirror_update_lower_bound - config.mirror_update_threshold) * 
				(float)clock / (float)config.iters_reach_lower_bound;
		}
		else
		{
			update_threshold = config.mirror_update_lower_bound;
		}
	}
  else 
  {
    /* By default, we apply the same strategy of adjusting learning rate to 
       mirror update thresholds */
    const string& lr_policy = config.lr_policy;
    if (lr_policy == "fixed") {
      thres_adj_factor = 1.0;
    } else if (lr_policy == "step") {
      int current_step = clock / config.lr_stepsize;
      thres_adj_factor = pow(config.lr_gamma, current_step);
    } else if (lr_policy == "exp") {
      thres_adj_factor = pow(config.lr_gamma, clock);
    } else if (lr_policy == "inv") {
      thres_adj_factor = pow(1.0 + config.lr_gamma * clock,
            - config.lr_power);
    } else if (lr_policy == "multistep") {
      int current_step = 0;
      for (int s = 0; s < config.lr_stepvalue.size(); s++) {
        if (clock >= config.lr_stepvalue[s]) {
          current_step += 1;
        }
      }
      thres_adj_factor = pow(config.lr_gamma, current_step);
    } else if (lr_policy == "poly") {
      thres_adj_factor = pow(1.0 - (float(clock) / float(config.max_iter)), 
                             config.lr_power);
    } else {
      /* Unknown policy */
      CHECK(0);
    } 
  }

  if (config.flush_mirror_update_per_iter != 0)
  {
    if (0 == clock % config.flush_mirror_update_per_iter)
    {
      update_threshold = 0.0;
    }
  }

  float update_val_threshold = config.mirror_update_value_threshold;

  if (config.lower_update_table_limit != 0)
  {
    if (table_id >= config.lower_update_table_limit)
    {
      update_val_threshold = config.lower_update_threshold;
    }
  }

  update_val_threshold *= thres_adj_factor;
  update_threshold *= thres_adj_factor;
  

	for (size_t v = 0; v < num_vals; v++) {
		
		if (accu_update[v] == 0 || config.local_model_only)
			continue;
    
		bool send_update = false;

    if (config.mirror_update_value_threshold != 0)
    {
      if (std::abs(accu_update[v]) > update_val_threshold)
      {
        send_update = true;
      }
    }
    else if (update_threshold >= 0)
    {        
      if (master_data[v] == 0)
      {
        send_update = true;
      }
      else
      {
        double update_ratio = 
          (double)accu_update[v] / (double)master_data[v];
        if (update_ratio < 0)
          update_ratio = -update_ratio;
        if (update_ratio > (double)update_threshold)
        {
          send_update = true;
        }
      }
    }

#if (ENABLE_UPDATE_STAT)
    {
      double update_ratio;
      if (master_data[v] == 0) {
        update_ratio = 1.0;
      } else {
        update_ratio = std::abs(accu_update[v] / (double)master_data[v]);
      }
      if (!isnan(update_ratio)) {  //NaN check
        server_stats.update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
        server_stats.update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        if (send_update) {
          server_stats.mirror_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
          server_stats.mirror_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        } 
        else {
          server_stats.local_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
          server_stats.local_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        }
      }
      if (!isnan(accu_update[v])) { //NaN check
        server_stats.update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
        server_stats.update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        if (send_update) {
          server_stats.mirror_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
          server_stats.mirror_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        } 
        else {
          server_stats.local_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
          server_stats.local_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        }
      }
    }
#endif

		if (send_update) 
		{				
			col_indexes->push_back(v);
			col_updates->push_back(accu_update[v]);
      if (!config.apply_change_to_local_model) {
        master_data[v] += accu_update[v];
      }
			accu_update[v] = 0;
			server_stats.total_mirror_update_send[clock/UPDATE_DIST_ITER_INTERVAL]++;
		}
	}

	return check_and_send_mirror_update(clock, table_id, 
																			col_indexes, col_updates);
}

/* The function to determine if we should send the mirror updates
	 We might queue the updates if we have used up our WAN BW */

bool TabletStorage::check_and_send_mirror_update(
	iter_t data_age, uint table_id, vector<col_idx_t>* col_indexes, 
	vector<val_t>* col_updates)
{
	/* This function is called ONLY when the mutex is held */
	
	DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;
  
	val_t *accu_update = reinterpret_cast<val_t *>(
			data_table.accu_update.data());
	val_t *inter_dcg_accu_update = reinterpret_cast<val_t *>(
			data_table.inter_dcg_accu_update.data());
		
	bool send_this_update = true;
	MirrorDest mirror_dest = MIRROR_DEST_ALL;
	if (config.enable_overlay_network)
	{
		mirror_dest = MIRROR_DEST_INTRA_DCG;
	}

	if (col_indexes->size())
	{
		bool send_col = true;
		size_t send_size = sizeof(col_idx_t) * col_indexes->size() + 
			sizeof(val_t) * col_updates->size();
		server_stats.total_mirror_update_batch++;

		/* Determine whether we should send selective columns or all columns */
		if ((sizeof(RowOpVal) * data_store.size()) <= send_size && config.apply_change_to_local_model)
		{
			send_col = false;
			send_size = sizeof(RowOpVal) * data_store.size();
			/* Adjust the stats */
			server_stats.total_mirror_update_send[data_age/UPDATE_DIST_ITER_INTERVAL] += 
				data_store.size() * ROW_DATA_SIZE - col_indexes->size();
		}
		else
		{
			server_stats.total_mirror_update_batch_col++;
		}

		/* Determine if we should send this update */

		if (config.enable_mirror_reorder && table_id != 0 &&
				last_mirror_send_size > 0)
		{
			/* Check how much time has elapsed since last sent */
			double time_from_last_send = 
				(tbb::tick_count::now() - last_mirror_send_time).seconds();

			/* Determine if last sent has finished */
			if (((last_mirror_send_size / time_from_last_send) > 
					wan_bandwidth_limit_byte_per_sec) || (table_id <= 2)) 
			{
				send_this_update = false;
			}
		}

		if (!send_this_update)
		{
			/* Store the info of this mirror update if we don't send it now */
			pending_mirror_updates.push_back(PendingMirrorUpdate());
			PendingMirrorUpdate& pending_update = 
				pending_mirror_updates[pending_mirror_updates.size() - 1];
			pending_update.init(data_age, table_id, col_indexes,
													col_updates, send_col, send_size);
			server_stats.total_mirror_update_pending_send++;
		}
		else 
		{
			/* Otherwise just send it now */
			last_mirror_send_time = tbb::tick_count::now();
			last_mirror_send_size = send_size;
			if (config.enable_threshold_by_bw)
			{
				if (top_layer_send_clock != data_age)
				{
					top_layer_mirror_send_time = tbb::tick_count::now();
					top_layer_send_clock = data_age;
					iter_mirror_send_size = 0;
				}
				iter_mirror_send_size += send_size;
			}

			if (send_col)
			{
				mc_communicator->send_mirror_update(
					data_age, table_id, *col_indexes, *col_updates, 
					reinterpret_cast<RowOpVal *>(accu_update), 
					data_store.size(), true, mirror_dest, false);			
				/* Save the updates to the inter-DCG buffer 
					 if it's a router */
				if (is_dcg_router)
				{
					for (size_t c = 0; c < col_indexes->size(); c++) {
						inter_dcg_accu_update[(*col_indexes)[c]] += (*col_updates)[c];
					}	
				}
			}
			else 
			{			
				/* Put back the significant updates, which we make 
					 them zero before */
				for (size_t c = 0; c < col_indexes->size(); c++) {
					accu_update[(*col_indexes)[c]] += (*col_updates)[c];
				}	
				/* Send out the updates */
				mc_communicator->send_mirror_update(
					data_age, table_id, *col_indexes, *col_updates, 
					reinterpret_cast<RowOpVal *>(accu_update), 
					data_store.size(), false, mirror_dest, false);
				/* Save the updates to the inter-DCG buffer 
					 if it's a router */
				if (is_dcg_router)
				{
					cpu_add(data_store.size() * ROW_DATA_SIZE,
									inter_dcg_accu_update,
									accu_update,
									inter_dcg_accu_update);
				}
				/* Zeroify the accu updates */
				data_table.accu_update.zerofy_data_cpu();
			}

			delete col_indexes;
			delete col_updates;

			server_stats.total_mirror_update_direct_send++;

			/* Send mirror clock here if we enable reodering */
			if (config.enable_mirror_reorder) {
				mc_communicator->send_mirror_clock(data_age, table_id,
																					 mirror_dest);
			}
		}
	}
	else 
	{
		delete col_indexes;
		delete col_updates;

		/* Send mirror clock here if we enable reodering */
		if (config.enable_mirror_reorder) {
			mc_communicator->send_mirror_clock(data_age, table_id,
																				 mirror_dest);
		}
	}

	/* If the table ID is 0, we check all the other pending 
		 mirror updates and send them here */
	if (config.enable_mirror_reorder && table_id == 0 &&
			pending_mirror_updates.size() > 0)
	{
		/* We always start from the back of the queue, as we assume they 
			 are more important */
		for (size_t i = pending_mirror_updates.size(); i > 0; i--)
		{
			PendingMirrorUpdate& pending_update = pending_mirror_updates[i - 1];
			if (!pending_update.initted)
				continue;

			DataTable& pend_data_table = data_tables[pending_update.table_id];
			DataStorage& pend_data_store = pend_data_table.store;
  
			val_t *pend_accu_update = reinterpret_cast<val_t *>(
				pend_data_table.accu_update.data());

			last_mirror_send_size += pending_update.send_size;

			if (pending_update.send_col)
			{
				mc_communicator->send_mirror_update(
					pending_update.data_age, pending_update.table_id, 
					*(pending_update.col_indexes), 
					*(pending_update.col_updates), 
					reinterpret_cast<RowOpVal *>(pend_accu_update), 
					pend_data_store.size(), true, mirror_dest, false);
			}
			else 
			{			
				/* Put back the significant updates, which we make 
					 them zero before */
				for (size_t c = 0; c < col_indexes->size(); c++) {
					pend_accu_update[(*pending_update.col_indexes)[c]] += (*pending_update.col_updates)[c];
				}	
				/* Send out the updates */
				mc_communicator->send_mirror_update(
					pending_update.data_age, pending_update.table_id, 
					*(pending_update.col_indexes), 
					*(pending_update.col_updates), 
					reinterpret_cast<RowOpVal *>(pend_accu_update), 
					pend_data_store.size(), false, mirror_dest, false);
				/* Zeroify the accu updates */
				pend_data_table.accu_update.zerofy_data_cpu();
			}
			delete pending_update.col_indexes;
			delete pending_update.col_updates;

			/* Send the mirror clock here */
			mc_communicator->send_mirror_clock(pending_update.data_age,
																				 pending_update.table_id,
																				 mirror_dest);
		}

		/* Clear up the queue */
		pending_mirror_updates.clear();
	}


	return send_this_update;
}


/* Send DCG mirror updates */
void TabletStorage::send_dcg_mirror_updates(uint table_id, iter_t clock, 
																						bool inter_dcg)
{
	/* This function is called ONLY when the mutex is held */

	CHECK_EQ(config.enable_gaia, 1);
	CHECK_EQ(config.enable_overlay_network, 1);

	DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = inter_dcg ? data_table.inter_dcg_accu_update :
		                                    data_table.intra_dcg_accu_update;

	size_t num_vals = data_store.size() * ROW_DATA_SIZE;
	val_t *dcg_update_data = reinterpret_cast<val_t *>(data_store.data());

	vector<col_idx_t> col_indexes;
	vector<val_t> col_updates;

	MirrorDest mirror_dest = inter_dcg ? MIRROR_DEST_INTER_DCG : MIRROR_DEST_INTRA_DCG;

	/* We send everything that is not zero */
	for (size_t v = 0; v < num_vals; v++) 
	{
		if (dcg_update_data[v] != 0)
		{
			col_indexes.push_back(v);
			col_updates.push_back(dcg_update_data[v]);
		}
	}

	if (col_indexes.size())
	{
		/* Determine if we should send cell or the whole table */
		size_t send_size = sizeof(col_idx_t) * col_indexes.size() + 
			sizeof(val_t) * col_updates.size();
		bool send_col = (sizeof(RowOpVal) * data_store.size()) > send_size;
		bool relay_dcg_update = (config.enable_olnw_multiple_routers && !inter_dcg);
	
		mc_communicator->send_mirror_update(
					clock, table_id, col_indexes, col_updates, 
					reinterpret_cast<RowOpVal *>(dcg_update_data), 
					data_store.size(), send_col, mirror_dest, relay_dcg_update);

		/* Zerofy the DCG updates */
		data_store.zerofy_data_cpu();
	}

}


/* Check and apply the model travelled here */
void TabletStorage::check_and_apply_travel_model(uint table_id, bool kick_clock)
{
  // We assume the mutex is held before invoking this function

  if (0 == config.model_traveling_freq)
  {
    return;
  }

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;
  val_t *master_data = reinterpret_cast<val_t *>(data_table.store.data());
  val_t *travel_data = reinterpret_cast<val_t *>(data_table.travel_model_store.data());  

  if (data_table.travel_arrive_clock > 0 &&
      0 == (data_table.travel_arrive_clock % config.model_traveling_freq) &&
      data_table.travel_arrive_clock == data_table.travel_leave_clock &&
      data_table.travel_apply_clock < data_table.travel_arrive_clock)
  {

#if (DEBUG_MODEL_TRAVEL)
      if (0 == channel_id && 0 == process_id)
			{
				LOG(INFO) << "Apply travelled model at clock: " << data_table.travel_arrive_clock
                  << ", table: " << table_id
                  << ", size: " << data_store.size();
			}
#endif        

    /* Copy the travelled model to the main store */
    cpu_copy(data_store.size() * ROW_DATA_SIZE,
             travel_data,
             master_data);

    /* Zeroify the accu updates */
    data_table.accu_update.zerofy_data_cpu();

    /* Update travel apply clock */
    data_table.travel_apply_clock = data_table.travel_arrive_clock;

    /* Reset travel arrive clock */
    data_table.travel_arrive_clock = 0;

    if (kick_clock)
    {
      /* Kick the local server to check the clock again
         We do this because the local server owns the socket
         to the local clients */
      communicator->send_local_clock_kick(table_id);
    }
  }
}

void TabletStorage::send_dgc_mirror_updates(uint table_id, iter_t clock) 
{

  /* This function is called ONLY when the mutex is held */

	CHECK_EQ(config.enable_dgc, 1);

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;
  
  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
  size_t num_vals = data_store.size() * ROW_DATA_SIZE;
	val_t *accu_update = reinterpret_cast<val_t *>(
    data_table.accu_update.data());
  val_t *update_history_data = reinterpret_cast<val_t *>(
    data_table.update_history.data());

	/* Send mirror updates if there are any significant ones */
	vector<col_idx_t> *col_indexes = new vector<col_idx_t>();
	vector<val_t> *col_updates = new vector<val_t>();

  float update_sparsity[5] = {0.75, 0.9375, 0.984375, 0.996, 0.999};

  int epoch = clock / config.dgc_epoch_size;
  float sparsity = 1.0;

  if (epoch < 4) {
    sparsity = 1.0 - update_sparsity[epoch];
  } else {
    sparsity = 1.0 - update_sparsity[4];
  }

  size_t sparse_update_num = (size_t)(num_vals * sparsity);
  if (num_vals > 0 && sparse_update_num == 0) {
    sparse_update_num = 1;
  }

  /* Determine the threshold of sparse updates */  
  std::priority_queue<val_t, std::vector<val_t>, std::greater<val_t> > pq;
  if (sparse_update_num > 0) {
    for (size_t v = 0; v < sparse_update_num; v++) {
      pq.push(std::abs(accu_update[v]));
    }
    for (size_t v = sparse_update_num; v < num_vals; v++) {
      val_t abs_update = std::abs(accu_update[v]);
      if (abs_update > pq.top()) { 
        pq.pop();
        pq.push(abs_update);
      }
    }
  }

  val_t sparse_update_thres = pq.empty() ? 0 : pq.top();

  for (size_t v = 0; v < num_vals; v++) {
    if (accu_update[v] == 0)
      continue;

#if (ENABLE_UPDATE_STAT)
    {
      double update_ratio;
      if (master_data[v] == 0) {
        update_ratio = 1.0;
      } else {
        update_ratio = std::abs(accu_update[v] / (double)master_data[v]);
      }
      if (!isnan(update_ratio)) {  //NaN check
        server_stats.update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
        server_stats.update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        if (std::abs(accu_update[v]) >= sparse_update_thres) {
          server_stats.mirror_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
          server_stats.mirror_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        } 
        else {
          server_stats.local_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][0] += update_ratio;
          server_stats.local_update_percent[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        }
      }
      if (!isnan(accu_update[v])) { //NaN check
        server_stats.update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
        server_stats.update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        if (std::abs(accu_update[v]) >= sparse_update_thres) {
          server_stats.mirror_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
          server_stats.mirror_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        } 
        else {
          server_stats.local_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][0] += std::abs(accu_update[v]);
          server_stats.local_update_magnitude[clock/UPDATE_DIST_ITER_INTERVAL][1] += 1;
        }
      }
    }
#endif

    
    if (std::abs(accu_update[v]) >= sparse_update_thres) {
      col_indexes->push_back(v);
			col_updates->push_back(accu_update[v]);
      if (!config.apply_change_to_local_model) {
        master_data[v] += accu_update[v];
      }
			accu_update[v] = 0;
      if (!config.disable_dgc_momemtum_mask) {
        update_history_data[v] = 0;  /*momemtum factor masking */
      }
			server_stats.total_mirror_update_send[clock/UPDATE_DIST_ITER_INTERVAL]++;
    }
  }  

  /* Send the updates */
  if (col_indexes->size())
  {
		server_stats.total_mirror_update_batch++;
		server_stats.total_mirror_update_batch_col++;

    /* Always send columns */
    mc_communicator->send_mirror_update(
      clock, table_id, *col_indexes, *col_updates, 
      reinterpret_cast<RowOpVal *>(accu_update), 
      data_store.size(), true, MIRROR_DEST_ALL, false);    
  }

  delete col_indexes;
  delete col_updates;

  return;
}

/* Get the DCG buffer based on the sender server ID */
val_t *TabletStorage::get_dcg_accu_buffer(uint table_id, uint from_server_id)
{
	CHECK(is_dcg_router);

	DataTable& data_table = data_tables[table_id];

	if (get_dcg_id(process_id) == get_dcg_id(from_server_id))
	{
		/* Update from the same DCG, save to the inter-DCG buffer */
		return reinterpret_cast<val_t *>(data_table.inter_dcg_accu_update.data());
	}
	else 
	{
		if (config.enable_olnw_multiple_routers)
		{
			CHECK_EQ(get_dc_id(from_server_id), config.dcg_peer_router_list[get_dc_id(process_id)]);
		}

		/* Update from another DCG, save to the intra-DCG buffer */
		return reinterpret_cast<val_t *>(data_table.intra_dcg_accu_update.data());
	}
}

/* Apply mirror updates for all columns */
void TabletStorage::apply_mirror_updates(
	uint table_id, val_t *update_vals, size_t num_vals, iter_t clock,
	uint from_server_id, uint relay_dcg_update)
{
	CHECK_EQ(config.enable_gaia || config.enable_dgc, 1);

	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

	CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;
  
  CHECK_EQ(data_store.size() * ROW_DATA_SIZE, num_vals);
  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());

#if (DEBUG_MODEL_TRAVEL)
      if (0 == channel_id && 0 == process_id)
			{
				LOG(INFO) << "Apply ALL mirror update at clock: " << clock
                  << ", table: " << table_id 
                  << ", size: " << data_store.size();
			}
#endif        

  cpu_add(num_vals,
					master_data,
					update_vals,
					master_data);

	if (is_dcg_router && !relay_dcg_update)
	{
		val_t *dcg_accu_data = get_dcg_accu_buffer(table_id, from_server_id);
		cpu_add(num_vals,
						dcg_accu_data,
						update_vals,
						dcg_accu_data);
	}
}

/* Apply mirror updates for selective columns */
void TabletStorage::apply_mirror_updates(
      uint table_id, col_idx_t* col_indexes, 
			val_t* col_updates, size_t col_num, iter_t clock,
			uint from_server_id, uint relay_dcg_update)
{

	CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];

	/* Make sure the table goes into clock 0, ugly workaround for now */
	volatile iter_t *table_clock =
		static_cast<volatile iter_t *>(&data_table.global_clock);
	while(*table_clock == INITIAL_DATA_AGE);

	CHECK_EQ(config.enable_gaia || config.enable_dgc, 1);

	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

  DataStorage& data_store = data_table.store;

#if 0
	if (table_id == 0 && channel_id == 0) {
		LOG(INFO) << "Received mirror update at server " << process_id
							<< " for " << col_num << " columns, "
							<< " first column " << col_indexes[0]
							<< ", value " << col_updates[0];
	}
#endif

#if (DEBUG_MODEL_TRAVEL)
      if (0 == channel_id && 0 == process_id)
			{
				LOG(INFO) << "Apply mirror update at clock: " << clock
                  << ", table: " << table_id 
                  << ", columns: " << col_num;
			}
#endif        

  
  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
  size_t num_vals = data_store.size() * ROW_DATA_SIZE;

	if (0 == num_vals) {
		LOG(INFO) << "Got mirror updates when data is 0 for clock " << clock
							<< " at server " << process_id
							<< " for table " << table_id
							<< " at channel " << channel_id;
	}
	CHECK_GT(col_num, 0);
	CHECK_LT(col_indexes[col_num - 1], num_vals);

	for (size_t c = 0; c < col_num; c++) {
		master_data[col_indexes[c]] += col_updates[c];
	}	

	if (is_dcg_router && !relay_dcg_update)
	{
		val_t *dcg_accu_data = get_dcg_accu_buffer(table_id, from_server_id);
		for (size_t c = 0; c < col_num; c++) {
			dcg_accu_data[col_indexes[c]] += col_updates[c];
		}	
	}
}


void TabletStorage::recv_model_travel(uint table_id, val_t *vals, size_t num_vals, 
                                      iter_t clock, uint from_server_id)
{
  CHECK_EQ(config.enable_gaia, 1);

	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

	CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& travel_model_store = data_table.travel_model_store;
  
  CHECK_EQ(travel_model_store.size() * ROW_DATA_SIZE, num_vals);
  val_t *travel_model_data = reinterpret_cast<val_t *>(travel_model_store.data());

  cpu_copy(num_vals,
           vals,
           travel_model_data);

  data_table.travel_arrive_clock = clock;

  check_and_apply_travel_model(table_id, true);
}


void TabletStorage::recv_model_value(uint table_id, val_t *vals, size_t num_vals, 
                                     iter_t clock, uint from_server_id)
{
  CHECK_EQ(config.enable_fedavg, 1);

	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

	CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  /* NOTE: We use travel model store to store the model values. So we cannot enable this
     with model traveling at the same time */
  DataStorage& travel_model_store = data_table.travel_model_store;
  
  CHECK_EQ(travel_model_store.size() * ROW_DATA_SIZE, num_vals);
  val_t *travel_model_data = reinterpret_cast<val_t *>(travel_model_store.data());

  cpu_add(num_vals,
          travel_model_data,
          vals,
          travel_model_data);
}

#if (ENABLE_UPDATE_DIST)
void TabletStorage::calc_update_dist(uint table_id, iter_t clock)
{
	CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;

	double update_threshold[UPDATE_DIST_COL_NUM] = 
		//{0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01};
		{0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001};

	val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
	size_t num_vals = data_store.size() * ROW_DATA_SIZE;

	for (int c = 0; c < UPDATE_DIST_COL_NUM; c++) 
  {
		val_t *shadow_update = reinterpret_cast<val_t *>(
			data_table.shadow_accu_update[c].data());
		
		// Go through all values and compare 
		for (size_t v = 0; v < num_vals; v++)
		{
			if (shadow_update[v] == 0)
				continue;

			bool send_update = false;
			if (master_data[v] == 0)
			{
				send_update = true;
			}
			else
			{
				float update_ratio = 
					shadow_update[v] / master_data[v];
				if (update_ratio < 0)
					update_ratio = -update_ratio;
				if (update_ratio > update_threshold[c])
				{
					send_update = true;
				}
			}
			if (send_update) 
			{
				shadow_update[v] = 0;
				server_stats.update_dist[table_id][clock/UPDATE_DIST_ITER_INTERVAL][c]++;
			}
		}
	}
}
#endif

void TabletStorage::process_multiclient_pending_reads(
	iter_t clock, uint table_id) {
  /* Rotate the starting client */
  uint client_to_start = clock % num_clients;
  /* Process pending reads */
  for (uint i = 0; i < num_clients; i++) {
    uint client_id = (client_to_start + i) % num_clients;
		if (config.enable_decentral) {
			/* Only push updates to the clients in the local servers
			   TODO: push updates to remote servers */
			if (get_dc_id(client_id) != get_dc_id(process_id)) {
				continue;
			}
		}
    process_pending_reads(client_id, clock, table_id);
  }
}

void TabletStorage::process_pending_reads(
	uint client_id, iter_t clock, uint table_id) {

  /* NOTE: we assume each client uses all the rows */
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  DataStorage& data_store = data_table.store;

  RowKeys& row_keys = data_store.row_keys;
  CHECK_EQ(data_store.size(), row_keys.size());
  RowData *row_data = data_store.data();
  CHECK_EQ(data_table.global_clock, clock);
  iter_t data_age = data_table.global_clock;
  iter_t self_clock = data_table.vec_clock[client_id];

#if (DEBUG_READ_TIME)
		if (client_id == 0)
		{
			LOG(INFO) << "SEND data to client from server: " << process_id
								<< ", for iter: " << clock
                << ", table: " << table_id
								<< ", channel: " << channel_id                
								<< ", at time: "
								<< (tbb::tick_count::now() - begin_time).seconds();
		}
#endif

#if 0
		if (0 == channel_id && (client_id - get_dc_id(client_id)) == 0)
		{
			LOG(INFO) << "SEND to client: " << client_id 
								<< ", from server: " << process_id
								<< ", table: " << table_id 
								<< ", iter: " << clock
								<< ", val: " << row_data[0].data[0]
								<< ", val: " << row_data[0].data[1]
								<< ", val: " << row_data[0].data[2]
								<< ", val: " << row_data[0].data[3]
								<< ", val: " << row_data[0].data[4];
		}
#endif

  communicator->read_row_batch_reply(
		client_id, process_id, data_age, self_clock, table_id,
		row_keys.data(), row_data, row_keys.size());
}

void TabletStorage::reset_perf_counters() {
  server_stats.reset();
}

void TabletStorage::clock(uint client_id, iter_t clock, uint table_id, 
													bool kick_clock) {
  int timing = true;
  tbb::tick_count clock_ad_start;
  tbb::tick_count clock_ad_apply_op_end;
  tbb::tick_count clock_ad_end;

	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

#if (DEBUG_READ_TIME)
		if (process_id == 0)
		{
			LOG(INFO) << "Server CLOCK for server: " << process_id
                << ", table: " << table_id
								<< ", channel: " << channel_id
								<< ", from client: " << client_id
								<< ", clock: " << clock
								<< ", kick_clock: " << kick_clock
								<< ", at time: "
								<< (tbb::tick_count::now() - begin_time).seconds();
		}
#endif

  if (timing) {
    clock_ad_start = tbb::tick_count::now();
  }

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
	MirrorDest mirror_dest = MIRROR_DEST_ALL;

	if (config.enable_overlay_network)
	{
		mirror_dest = MIRROR_DEST_INTRA_DCG;
	}

	CHECK_LT(client_id, data_table.vec_clock.size());

	if (!kick_clock) {		
		if (data_table.vec_clock[client_id] != INITIAL_DATA_AGE) {
			CHECK_EQ(clock, data_table.vec_clock[client_id] + 1);
		}
		data_table.vec_clock[client_id] = clock;
	} 

  iter_t new_global_clock = 0;
	if (config.enable_decentral) {

		iter_t new_local_clock = 0;
		if (data_table.local_clock != INITIAL_DATA_AGE) {
			new_local_clock = local_clock_min(data_table.vec_clock);
		} else {
			/* At the beginning the initial values are pushed by 
				 worker 0, so the first local clock needs to check all 
				 client */
			new_local_clock = clock_min(data_table.vec_clock);
		}

		if (new_local_clock != data_table.local_clock) 
    {
			data_table.local_clock = new_local_clock;

      /* Decentralized ML case, we apply local updates to master store 
         when we have received updates from all local clients */
      apply_local_updates_to_store(table_id, new_local_clock);

      /* Gaia case, send mirror updates to other DCs */
      if (config.enable_gaia)
      {
        /* The initial values are pushed by worker 0, otherwise
           we send mirror updates if we want to merge them here*/
        bool send_mirror_clock = true;
        if (config.merge_local_update && new_local_clock != 0) {
          if (!send_mirror_updates(table_id, new_local_clock)) {
            send_mirror_clock = false;
          }
        }

        /* Send mirror clock to other DCs */
        if (send_mirror_clock && 
            (!config.enable_mirror_reorder || new_local_clock == 0)) 
        {
#if (DEBUG_READ_TIME)
          if (process_id == 0)
          {
            LOG(INFO) << "SEND mirror clock from server: " << process_id
                      << " for mc: " << data_table.local_clock
                      << ", table: " << table_id
                      << ", channel: " << channel_id
                      << ", at time: "
                      << (tbb::tick_count::now() - begin_time).seconds();
          }
#endif
          mc_communicator->send_mirror_clock(data_table.local_clock,
                                             table_id, mirror_dest);
        }
      }

      /* FedAvg case, send model value after local iterations */
      if (config.enable_fedavg)
      {        
        if (new_local_clock > 0 && (0 == new_local_clock % config.fedavg_local_iter))
        {
          DataStorage& data_store = data_table.store;
          mc_communicator->send_model_value(data_table.local_clock,
                                            table_id,
                                            reinterpret_cast<RowOpVal *>(data_store.data()),
                                            data_store.size(),
                                            MIRROR_DEST_ALL);
        }
        /* Here we reuse the mirror clock for Gaia to synchronize FedAvg. We can further
           save this part by only sending/checking mirror clocks when we need to */
        mc_communicator->send_mirror_clock(data_table.local_clock,
                                           table_id, MIRROR_DEST_ALL);
      }

      /* DGC case, determine sparse updates and send them to other DCs
         Also apply sparse updates to the local model */
      if (config.enable_dgc)
      {
        if (new_local_clock > 0) 
        {
          send_dgc_mirror_updates(table_id, new_local_clock);          
        }

        /* Reuse Gaia mirror clock for synchronization */
        mc_communicator->send_mirror_clock(data_table.local_clock,
                                           table_id, MIRROR_DEST_ALL);
      }
		}

		if (config.enable_overlay_network && config.enable_gaia)
		{
			/* Check intra DCG clock */
			iter_t new_intra_dcg_clock = intra_dcg_clock_min(
				data_table.local_clock, data_table.mirror_clock);

			if (data_table.dcg_clock[get_dcg_id(process_id)] !=
					new_intra_dcg_clock)
			{
				data_table.dcg_clock[get_dcg_id(process_id)] = new_intra_dcg_clock;
				if (is_dcg_router)
				{
					if (new_intra_dcg_clock != 0)
					{
						/* Send inter-DCG updates to other DCGs */
						send_dcg_mirror_updates(table_id, new_intra_dcg_clock,
																		true);
					}

					/* Send mirror clock to other DCGs */
					mc_communicator->send_mirror_clock(new_intra_dcg_clock,
																						 table_id, 
																						 MIRROR_DEST_INTER_DCG);
				}
			}

			/* Check inter DCG clock */
			iter_t new_inter_dcg_clock;
			if (config.enable_olnw_multiple_routers && is_dcg_router)
			{
				new_inter_dcg_clock = data_table.dcg_clock[
					get_dcg_id(config.dcg_peer_router_list[get_dc_id(process_id)])];
			}
			else 
			{
				new_inter_dcg_clock = clock_min(data_table.dcg_clock);
			}
			
			if (new_inter_dcg_clock != data_table.inter_dcg_clock)
			{
				data_table.inter_dcg_clock = new_inter_dcg_clock;
				if (is_dcg_router)
				{
					if (new_inter_dcg_clock != 0) 
					{
						/* Send intra-DCG updates to servers within this DCG */
						send_dcg_mirror_updates(table_id, new_intra_dcg_clock,
																		false);
					}

					/* Send inter-DCG clock to servers within this DCG */
					mc_communicator->send_inter_dcg_clock(new_inter_dcg_clock,
																								table_id);
				}
			}
		}

    // !!! XXX: we assume there are only two DCs, so mirror clock ticks
    // !!! the model traveling. We need to design another way to do this
    // !!! when we have multiple DCs
    if (0 != config.model_traveling_freq && config.enable_gaia)
    {
      iter_t new_travel_leave_clock = 0;
      new_travel_leave_clock = travel_leave_clock_min(data_table.local_clock, 
                                                      data_table.mirror_clock,
                                                      table_id);
      if (new_travel_leave_clock != data_table.travel_leave_clock && 
          new_travel_leave_clock > 0 &&
          0 == new_travel_leave_clock % config.model_traveling_freq)
      {
        data_table.travel_leave_clock = new_travel_leave_clock;
        DataStorage& data_store = data_table.store;
        mc_communicator->send_model_travel(new_travel_leave_clock, 
                                           table_id,
                                           reinterpret_cast<RowOpVal *>(data_store.data()),
                                           data_store.size(),
                                           MIRROR_DEST_ALL); //!! Need to change for multiple DCs

#if (DEBUG_MODEL_TRAVEL)
        if (0 == channel_id && 0 == process_id)
        {
          LOG(INFO) << "Send model out at clock: " << new_travel_leave_clock
                    << ", table: " << table_id
                    << ", size: " << data_store.size();
        }
#endif        

        check_and_apply_travel_model(table_id, false);
      }
    }		

		new_global_clock = global_clock_min(data_table.local_clock, 
																				data_table.mirror_clock,
																				data_table.dcg_clock,
                                        data_table.travel_apply_clock,
                                        data_table.global_clock,
																				table_id);

  } else {
		new_global_clock = clock_min(data_table.vec_clock);
		/* We don't check local clock for non-Gaia case */
  }
  
  if (new_global_clock != data_table.global_clock) {
    if (data_table.global_clock != INITIAL_DATA_AGE) {
      CHECK_EQ(new_global_clock, data_table.global_clock + 1);
    }
    data_table.global_clock = new_global_clock;

#if (HISTORY_IN_SERVER)
    /* Non-decentralized case, we apply local updates to master store when 
       the global clock moves */
    if (!config.enable_decentral) {
      apply_local_updates_to_store(table_id, new_global_clock);
    }
#endif   

#if (DEBUG_MODEL_TRAVEL)
    if (0 == channel_id && 0 == process_id)
    {
      LOG(INFO) << "Send pending read request at clock: " << data_table.global_clock
                << ", table: " << table_id;
    }
#endif

    /* FedAvg case, we average weights among all DCs before sending them back
       to the clients */
    if (config.enable_fedavg && data_table.global_clock > 0 
        && (0 == data_table.global_clock % config.fedavg_local_iter))
    {
      apply_fed_avg_weights(table_id);
    }

    /* Send pending read requests */
    process_multiclient_pending_reads(
			data_table.global_clock, table_id);

    /* Notify clients of new iteration */
    /* We don't need to do that now, because the client will use
     * the reception of row data as the notification. */
    // communicator->clock(process_id, global_clock);
  }

  if (timing) {
    clock_ad_end = tbb::tick_count::now();
    server_stats.clock_ad_send_pending_time +=
      (clock_ad_end - clock_ad_apply_op_end).seconds();
    server_stats.clock_ad_time_tot +=
      (clock_ad_end - clock_ad_start).seconds();
  }
}

void TabletStorage::mirror_clock(uint server_id, iter_t clock, 
																 uint table_id) 
{
	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

	CHECK_EQ(config.enable_decentral, 1);
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];

	uint dc_id = get_dc_id(server_id);
	if (data_table.mirror_clock[dc_id] != INITIAL_DATA_AGE) {
		CHECK_EQ(clock, data_table.mirror_clock[dc_id] + 1);
	}
	data_table.mirror_clock[dc_id] = clock;

#if (DEBUG_READ_TIME)
		if (process_id == 0)
		{
			LOG(INFO) << "Server RECV mirror clock at server: " << process_id
                << ", table: " << table_id
								<< ", channel: " << channel_id
								<< ", clock: " << clock
								<< ", at time: "
								<< (tbb::tick_count::now() - begin_time).seconds();
		}
#endif

	if (is_dcg_router)
	{
		/* Check if it's from another DCG */
		uint dcg_id = get_dcg_id(server_id);
		if (dcg_id != get_dcg_id(process_id))
		{
			if (data_table.dcg_clock[dcg_id] != INITIAL_DATA_AGE) {
				CHECK_EQ(clock, data_table.dcg_clock[dcg_id] + 1);
			}
			data_table.dcg_clock[dcg_id] = clock;
		}
	}

  // !!! XXX: we assume there are only two DCs, so mirror clock ticks
  // !!! the model traveling. We need to design another way to do this
  // !!! when we have multiple DCs
  if (0 != config.model_traveling_freq)
  {
    iter_t new_travel_leave_clock = 0;
    new_travel_leave_clock = travel_leave_clock_min(data_table.local_clock, 
                                                    data_table.mirror_clock,
                                                    table_id);
    if (new_travel_leave_clock != data_table.travel_leave_clock && 
        new_travel_leave_clock > 0 &&
        0 == new_travel_leave_clock % config.model_traveling_freq)
    {
      data_table.travel_leave_clock = new_travel_leave_clock;
      DataStorage& data_store = data_table.store;
      mc_communicator->send_model_travel(new_travel_leave_clock, 
                                         table_id,
                                         reinterpret_cast<RowOpVal *>(data_store.data()),
                                         data_store.size(),
                                         MIRROR_DEST_ALL); //!! Need to change for multiple DCs

#if (DEBUG_MODEL_TRAVEL)
      if (0 == channel_id && 0 == process_id)
			{
				LOG(INFO) << "Send model out at clock: " << new_travel_leave_clock
                  << ", table: " << table_id
                  << ", size: " << data_store.size();
			}
#endif      

      check_and_apply_travel_model(table_id, false);
    }
  }

  iter_t new_global_clock = 0;
  new_global_clock = global_clock_min(data_table.local_clock, 
                                      data_table.mirror_clock,
                                      data_table.dcg_clock,
                                      data_table.travel_apply_clock,
                                      data_table.global_clock,
                                      table_id);
  
  if (new_global_clock != data_table.global_clock ||
      config.enable_overlay_network) {

    /* Kick the local server to check the clock again
       We do this because the local server owns the socket
       to the local clients */
    communicator->send_local_clock_kick(table_id);
  }
}


void TabletStorage::inter_dcg_clock(uint server_id, iter_t clock, 
																		uint table_id)
{
	boost::unique_lock<mutex> tablet_lock(*tablet_mutex);

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];

	CHECK(config.enable_overlay_network);

	if (config.enable_olnw_multiple_routers)
  {
		/* If multiple routers, we only update the DCG clock that is 
			 from the peer router of the sender */
		CHECK(config.dcg_router_list[get_dc_id(server_id)]);
		uint dcg_id = get_dcg_id(config.dcg_peer_router_list[get_dc_id(server_id)]);
		if (data_table.dcg_clock[dcg_id] != INITIAL_DATA_AGE) {
			CHECK_EQ(clock, data_table.dcg_clock[dcg_id] + 1);
		}
		data_table.dcg_clock[dcg_id] = clock;
	}
	else
	{
		/* Must be a non-router */
		CHECK(!is_dcg_router);

		/* Update all the other DCG clocks */
		for (uint dcg_id = 0; dcg_id < data_table.dcg_clock.size(); dcg_id++)
		{
			if (dcg_id != get_dcg_id(process_id)) {
				if (data_table.dcg_clock[dcg_id] != INITIAL_DATA_AGE) {
					CHECK_EQ(clock, data_table.dcg_clock[dcg_id] + 1);
				}
				data_table.dcg_clock[dcg_id] = clock;
			}
		}
	}

	/* Kick the local server to check the clock again
		 We do this because the local server owns the socket
		 to the local clients */
	communicator->send_local_clock_kick(table_id);
}

void TabletStorage::get_stats(
      uint client_id, shared_ptr<MetadataServer> metadata_server) {
  server_stats.nr_rows = 0;
  for (uint table_id = 0; table_id < data_tables.size(); table_id++) {
    server_stats.nr_rows += data_tables[table_id].row_count;
  }

  std::stringstream combined_server_stats;
  if (config.enable_decentral) {
    combined_server_stats << "{"
                          << "\"storage\": " << server_stats.to_json() << ", "
                          << "\"metadata\": " << metadata_server->get_stats() << ", "
                          << "\"router\": " << communicator->get_router_stats() << ", "
                          << "\"mirror_client_router\": " << mc_communicator->get_mc_router_stats() << ", " 
                          << "\"mirror_server_router\": " << mc_communicator->get_ms_router_stats()
                          << " } ";
  } else {
    combined_server_stats << "{"
                          << "\"storage\": " << server_stats.to_json() << ", "
                          << "\"metadata\": " << metadata_server->get_stats() << ", "
                          << "\"router\": " << communicator->get_router_stats() 
                          << " } ";
  }
  communicator->get_stats(client_id, combined_server_stats.str());
}

void TabletStorage::param_table_offset(const std::vector<ParamTableRowOffset>& param_offset_info) {
  for (uint param_idx = 0; param_idx < param_offset_info.size(); param_idx++) {
    uint table_id = param_offset_info[param_idx].table_id;
    CHECK_LT(table_id, data_tables.size());
    data_tables[table_id].param_offset_list.push_back(param_offset_info[param_idx]);
  }
}

void TabletStorage::param_learning_rate(int global_param_id, float learning_rate) {
  boost::unique_lock<mutex> tablet_lock(*tablet_mutex);
  
  param_learning_rates[global_param_id] = learning_rate;

  //LOG(INFO) << "Set Param " << global_param_id << ", LR = " << learning_rate;
}

void TabletStorage::param_momentum(int global_param_id, float momentum) {
  boost::unique_lock<mutex> tablet_lock(*tablet_mutex);
  
  param_momentums[global_param_id] = momentum;

  //LOG(INFO) << "Set Param " << global_param_id << ", Momentum = " << momentum;
}

uint TabletStorage::get_dc_id(uint machine_id) {
	return machine_id % config.num_dc;
}

uint TabletStorage::get_dcg_id(uint machine_id) {
	return config.dcg_id_list[get_dc_id(machine_id)];
}

iter_t TabletStorage::local_clock_min(vector<iter_t> clocks) {
	CHECK(clocks.size());
  iter_t cmin = clocks[get_dc_id(process_id)];
  for (uint i = 0; i < clocks.size(); i++) {
		if (get_dc_id(i) == get_dc_id(process_id)) {
			cmin = clocks[i] < cmin ? clocks[i] : cmin;
		}
  }
  return cmin;
}

iter_t TabletStorage::intra_dcg_clock_min(iter_t local_clock, 
																					vector<iter_t> mirror_clocks) 
{
	CHECK(mirror_clocks.size());
  iter_t cmin = MAX_CLOCK;

	for (uint i = 0; i < mirror_clocks.size(); i++) {
		if (i != get_dc_id(process_id) && 
				get_dcg_id(i) == get_dcg_id(process_id)) 
		{
			cmin = mirror_clocks[i] < cmin ? mirror_clocks[i] : cmin;
		}
  }

	/* TODO: Add slack for intra DC clock? */

	cmin = local_clock < cmin ? local_clock : cmin;
	
  return cmin;
}

iter_t TabletStorage::travel_leave_clock_min(iter_t local_clock, 
                                             vector<iter_t> mirror_clocks,
                                             uint table_id){
	CHECK(mirror_clocks.size());
  iter_t cmin = MAX_CLOCK;

  for (uint i = 0; i < mirror_clocks.size(); i++) 
  {
    if (i != get_dc_id(process_id)) 
    {
      cmin = mirror_clocks[i] < cmin ? mirror_clocks[i] : cmin;
    }
  }

	cmin = local_clock < cmin ? local_clock : cmin;
	
  return cmin;
}


iter_t TabletStorage::global_clock_min(iter_t local_clock, 
                                       vector<iter_t> mirror_clocks,
                                       vector<iter_t> dcg_clocks,
                                       iter_t travel_apply_clock,
                                       iter_t old_global_clock,
                                       uint table_id){
	CHECK(mirror_clocks.size());
  iter_t cmin = MAX_CLOCK;

	if (config.enable_overlay_network)
  {
		for (uint i = 0; i < dcg_clocks.size(); i++) {
			if (i != get_dcg_id(process_id)) {
				cmin = dcg_clocks[i] < cmin ? dcg_clocks[i] : cmin;
			}
		}
		if (cmin != INITIAL_DATA_AGE && config.slack_table_limit >= 0 &&
				table_id < (uint)config.slack_table_limit) {		
			/* Increase a slack for the mirror clocks */
			cmin += 1;
		}
		cmin = dcg_clocks[get_dcg_id(process_id)] < cmin ? dcg_clocks[get_dcg_id(process_id)] : cmin;
	}
	else
	{
		for (uint i = 0; i < mirror_clocks.size(); i++) {
			if (i != get_dc_id(process_id)) {
				cmin = mirror_clocks[i] < cmin ? mirror_clocks[i] : cmin;
        //if (mirror_clocks[i] == INITIAL_DATA_AGE) {
        //  LOG(INFO) << "Mirror clock " << i << " is at INITIAL_DATA_AGE";
        //}
			}
		}
		if (cmin != INITIAL_DATA_AGE && config.slack_table_limit >= 0 &&
				table_id < (uint)config.slack_table_limit) 
    {		
        /* Increase a slack for the mirror clocks */
        cmin += 1; 
		}
	}

	cmin = local_clock < cmin ? local_clock : cmin;

  if (0 != config.model_traveling_freq &&
      cmin > 0 &&
      0 == (cmin % config.model_traveling_freq) &&
      travel_apply_clock < cmin)
  {
    cmin = old_global_clock;
  }
	
  return cmin;
}
