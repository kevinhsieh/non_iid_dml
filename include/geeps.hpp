#ifndef __geeps_hpp__
#define __geeps_hpp__

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

#include <string>
#include <vector>

#include "geeps-user-defined-types.hpp"

using std::string;
using std::vector;

struct GeePsConfig {
  uint num_tables;
  std::vector<std::string> host_list;
  std::vector<uint> port_list;
  uint tcp_base_port;
  uint num_comm_channels;
  std::string output_dir;
  iter_t log_interval;
  int pp_policy;
  int local_opt;
  size_t gpu_memory_capacity;
  int mm_warning_level;
      /* 0: no warning
       * 1: guarantee double buffering for thread cache
       * 2: make sure all local data in GPU memory
       * 3: make sure all parameter cache in GPU memory */
  int pinned_cpu_memory;
  int read_my_writes;
	int enable_gaia;
  int num_dc;
	float mirror_update_threshold;
	int merge_local_update;
	int aggr_mirror_update_table_group;
	float aggr_mirror_update_threshold;
	int enable_mirror_reorder;
	float wan_bandwidth_limit;	
  int slack_table_limit;
	float mirror_update_lower_bound;
  int iters_reach_lower_bound;
	int enable_overlay_network;
	int num_dcg;
	std::vector<uint> dcg_id_list;
	std::vector<bool> dcg_router_list;
	int enable_olnw_multiple_routers;
	std::vector<uint> dcg_peer_router_list;
	int enable_threshold_by_bw;
  int flush_mirror_update_per_iter;
  int local_model_only;
  float mirror_update_value_threshold;
  float conf_ratio_to_lower_update_threshold;
  float lower_update_threshold;
  int lower_update_table_limit;
  int model_traveling_freq;
  /* Parameters to determine the policy of threshold */
  std::string lr_policy;
  int max_iter;
  float lr_gamma;
  float lr_power;
  int lr_stepsize;
  std::vector<int> lr_stepvalue;
  float momentum;
  int enable_fedavg;                /* FedAvg algorithm */
  int fedavg_local_iter;            /* The number of local iterations for FedAvg */
  int enable_decentral;             /* Decentralized training */
  int enable_dgc;                   /* Deep gradient compression */
  int dgc_epoch_size;               /* The number iters per epoch for DGC warm-up */
  int apply_change_to_local_model;  /* Apply local changes to local model for each data center */
  int disable_dgc_momemtum_mask;    /* Disable momemtum factor masking in DGC */
  std::vector<float> gaia_threshold_sched;
  int gaia_threshold_iter;
  int gradient_clip_threshold;

  GeePsConfig() :
    num_tables(1),
    tcp_base_port(9090),
    num_comm_channels(1),
    output_dir(""), log_interval(0),
    pp_policy(0), local_opt(1),
    gpu_memory_capacity(std::numeric_limits<size_t>::max()),
    mm_warning_level(1),
    pinned_cpu_memory(1),
    read_my_writes(0),
    enable_gaia(0),
    num_dc(1),
		mirror_update_threshold(0),
		merge_local_update(0),
		aggr_mirror_update_table_group(0),
		aggr_mirror_update_threshold(0), 
		enable_mirror_reorder(0), 
		wan_bandwidth_limit(100.0),
		slack_table_limit(-1),
		mirror_update_lower_bound(0), 
		iters_reach_lower_bound(-1),
    enable_overlay_network(0),
		num_dcg(1),
		enable_olnw_multiple_routers(0),
		enable_threshold_by_bw(0),
    flush_mirror_update_per_iter(0),
    local_model_only(0),
    mirror_update_value_threshold(0),
    lower_update_threshold(0),
    lower_update_table_limit(0),
    model_traveling_freq(0),
    max_iter(0), 
    lr_gamma(1.0), 
    lr_power(1.0),
    lr_stepsize(1000),
    momentum(0.9),
    enable_fedavg(0),
    fedavg_local_iter(1),
    enable_decentral(0),
    enable_dgc(0),
    dgc_epoch_size(1000),
    apply_change_to_local_model(0),
    disable_dgc_momemtum_mask(0),
    gradient_clip_threshold(0)
    {}
};

class GeePs {
 public:
  GeePs(uint process_id, const GeePsConfig& config);
  void Shutdown();
  std::string GetStats();
  void StartIterations();

  /* Interfaces for virtual iteration */
  int VirtualRead(size_t table_id, const vector<size_t>& row_ids, int slack);
  int VirtualPostRead(int prestep_handle);
  int VirtualPreUpdate(size_t table_id, const vector<size_t>& row_ids,
                       const vector<int>& global_param_ids,
                       const vector<int>& val_offsets);
  int VirtualUpdate(int prestep_handle);
  int VirtualLocalAccess(const vector<size_t>& row_ids, bool fetch);
  int VirtualPostLocalAccess(int prestep_handle, bool keep);
  int VirtualClock();
  void FinishVirtualIteration();

  /* Interfaces for real access */
  bool Read(int handle, RowData **buffer_ptr);
  void PostRead(int handle);
  void PreUpdate(int handle, RowOpVal **buffer_ptr);
  void Update(int handle);
  bool LocalAccess(int handle, RowData **buffer_ptr);
  void PostLocalAccess(int handle);
  void SetParamLearningRate(int param_id, float learning_rate);
  void SetParamMomentum(int param_id, float momentum);
  void Clock();
};

#define HISTORY_IN_SERVER (1)

#endif  // defined __geeps_hpp__
