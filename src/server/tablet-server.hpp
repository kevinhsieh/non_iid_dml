#ifndef __tablet_server_hpp__
#define __tablet_server_hpp__

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

#include <tbb/tick_count.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <map>
#include <cmath>

#include "geeps-user-defined-types.hpp"
#include "common/common-util.hpp"
#include "common/row-op-util.hpp"
#include "server-encoder-decoder.hpp"
#include "mirror-server-encoder-decoder.hpp"
#include "metadata-server.hpp"

using std::string;
using std::vector;
using std::set;
using boost::shared_ptr;
using boost::unique_lock;
using boost::mutex;

//#define UPDATE_DIST_COL_NUM (15)
#define UPDATE_DIST_COL_NUM (9)
#define UPDATE_DIST_TABLE_NUM (13)
#define UPDATE_DIST_ITER_NUM (700)
#define UPDATE_DIST_ITER_INTERVAL (500)
#define ENABLE_UPDATE_DIST (0)

#define ENABLE_UPDATE_STAT (1)

class ServerClientEncode;
class MirrorClientEncode;
class MetadataServer;   /* Used in get_stats() */

class TabletStorage {
 public:
  struct Stats {
    int64_t nr_request;
    int64_t nr_request_prior;
    int64_t nr_local_request;
    int64_t nr_send;
    int64_t nr_update;
    int64_t nr_local_update;
    int64_t nr_rows;
    double send_data_time;
    double clock_ad_apply_op_time;
    double clock_ad_send_pending_time;
    double clock_ad_time_tot;
    double iter_var_time;
    double inc_time;

#if (ENABLE_UPDATE_DIST)
    //int64_t update_dist[MAX_ITER][UPDATE_DIST_COL_NUM];
		int64_t update_dist[UPDATE_DIST_TABLE_NUM][UPDATE_DIST_ITER_NUM][UPDATE_DIST_COL_NUM];
#endif

#if (ENABLE_UPDATE_STAT)
    double update_percent[UPDATE_DIST_ITER_NUM][2];
    double mirror_update_percent[UPDATE_DIST_ITER_NUM][2];
    double local_update_percent[UPDATE_DIST_ITER_NUM][2];
    double update_magnitude[UPDATE_DIST_ITER_NUM][2];
    double mirror_update_magnitude[UPDATE_DIST_ITER_NUM][2];
    double local_update_magnitude[UPDATE_DIST_ITER_NUM][2];
#endif

		int64_t total_update[UPDATE_DIST_ITER_NUM];
		int64_t total_mirror_update_send[UPDATE_DIST_ITER_NUM];
		int64_t total_mirror_update_batch;
		int64_t total_mirror_update_batch_col;
		int64_t total_mirror_update_direct_send;
		int64_t total_mirror_update_pending_send;

		double total_update_threshold_adj[UPDATE_DIST_ITER_NUM];

    int64_t total_grad_clip_check;
    int64_t total_grad_clip_apply;
    double max_grad_l2_norm;


    void reset() {
      nr_request = 0;
      nr_request_prior = 0;
      nr_local_request = 0;
      nr_send = 0;
      nr_update = 0;
      nr_local_update = 0;
      send_data_time = 0.0;
      clock_ad_time_tot = 0.0;
      clock_ad_apply_op_time = 0.0;
      clock_ad_send_pending_time = 0.0;
      inc_time = 0.0;
      iter_var_time = 0.0;

#if (ENABLE_UPDATE_DIST)
			for (int t = 0; t < UPDATE_DIST_TABLE_NUM; t++) {
				for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
					for (int c = 0; c < UPDATE_DIST_COL_NUM; c++) {
						update_dist[t][i][c] = 0;
					}
				}
			}			
#endif

#if (ENABLE_UPDATE_STAT)
      for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
        for (int j = 0; j < 2; j++) {
          update_percent[i][j] = 0;
          mirror_update_percent[i][j] = 0;
          local_update_percent[i][j] = 0;
          update_magnitude[i][j] = 0;
          mirror_update_magnitude[i][j] = 0;
          local_update_magnitude[i][j] = 0;
        }
      }
#endif

			for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
				total_update_threshold_adj[i] = 0;
				total_update[i] = 0;
				total_mirror_update_send[i] = 0;
			}

			total_mirror_update_batch = 0;
			total_mirror_update_batch_col = 0;
			total_mirror_update_direct_send = 0;
			total_mirror_update_pending_send = 0;
      total_grad_clip_check = 0;
      total_grad_clip_apply = 0;
      max_grad_l2_norm = 0.0;
    }

    Stats() {
      reset();
    }

    Stats& operator += (const Stats& rhs) {
      return *this;
    }
    std::string to_json() {
      std::stringstream ss;
      ss << "{"
         << "\"nr_rows\": " << nr_rows << ", "
         << "\"nr_request\": " << nr_request << ", "
         << "\"nr_request_prior\": " << nr_request_prior << ", "
         << "\"nr_local_request\": " << nr_local_request << ", "
         << "\"nr_send\": " << nr_send << ", "
         << "\"nr_update\": " << nr_update << ", "
         << "\"nr_local_update\": " << nr_local_update << ", "
         << "\"send_data_time\": " << send_data_time << ", "
         << "\"clock_ad_apply_op_time\": " << clock_ad_apply_op_time << ", "
         << "\"clock_ad_send_pending_time\": "
         << clock_ad_send_pending_time << ", "
         << "\"clock_ad_time_tot\": " << clock_ad_time_tot << ", "
         << "\"iter_var_time\": " << iter_var_time << ", ";

#if (ENABLE_UPDATE_DIST)
			for (int t = 0; t < UPDATE_DIST_TABLE_NUM; t++) {
				for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
					for (int c = 0; c < UPDATE_DIST_COL_NUM; c++) {
						ss << "\"update_dist[" << t << "][" << i << "][" << c 
							 << "]\": " << update_dist[t][i][c] << ", ";
					}
				}
			}			
#endif

#if (ENABLE_UPDATE_STAT)
      for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
        ss << "\"update_percent[" << i << "]\":"
           << ((update_percent[i][1] == 0.0) ? 0.0 :
               (update_percent[i][0] / update_percent[i][1])) << ", ";
        ss << "\"mirror_update_percent[" << i << "]\":"
           << ((mirror_update_percent[i][1] == 0.0) ? 0.0 :
               (mirror_update_percent[i][0] / mirror_update_percent[i][1])) << ", ";
        ss << "\"local_update_percent[" << i << "]\":"
           << ((local_update_percent[i][1] == 0.0) ? 0.0 :
               (local_update_percent[i][0] / local_update_percent[i][1])) << ", ";
        ss << "\"update_magnitude[" << i << "]\":"
           << ((update_magnitude[i][1] == 0.0) ? 0.0 :
               (update_magnitude[i][0] / update_magnitude[i][1])) << ", ";
        ss << "\"mirror_update_magnitude[" << i << "]\":"
           << ((mirror_update_magnitude[i][1] == 0.0) ? 0.0 :
               (mirror_update_magnitude[i][0] / mirror_update_magnitude[i][1])) << ", ";
        ss << "\"local_update_magnitude[" << i << "]\":"
           << ((local_update_magnitude[i][1] == 0.0) ? 0.0 :
               (local_update_magnitude[i][0] / local_update_magnitude[i][1])) << ", ";
      }
#endif

			for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
				ss << "\"total_update_threshold_adj[" << i << "]\":"
					 << total_update_threshold_adj[i] << ", ";
			}

			for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
				ss << "\"total_update[" << i << "]\":"
					 << total_update[i] << ", ";
			}

			for (int i = 0; i < UPDATE_DIST_ITER_NUM; i++) {
				ss << "\"total_mirror_update_send[" << i << "]\":"
					 << total_mirror_update_send[i] << ", ";
			}

			ss << "\"total_mirror_update_batch\": " << total_mirror_update_batch << ", "
				 << "\"total_mirror_update_batch_col\": " << total_mirror_update_batch_col << ", "
				 << "\"total_mirror_update_direct_send\": " << total_mirror_update_direct_send << ", "
				 << "\"total_mirror_update_pending_send\": " << total_mirror_update_pending_send << ", "
         << "\"total_grad_clip_check\": " << total_grad_clip_check << ", "
         << "\"total_grad_clip_apply\": " << total_grad_clip_apply << ", "
         << "\"max_grad_l2_norm\": " << max_grad_l2_norm << ", "
         << "\"inc_time\": " << inc_time
         << " } ";
      return ss.str();
    }
  };
  Stats server_stats;

  typedef boost::unordered_map<TableRow, uint> Row2Index;

  typedef boost::unordered_map<iter_t, RowKeys> PendingReadsLog;
  typedef std::vector<PendingReadsLog> MulticlientPendingReadsLog;
  typedef std::vector<iter_t> VectorClock;
  typedef std::vector<ParamTableRowOffset> ParamOffset;
  typedef boost::unordered_map<int, float> ParamLearningRate;
  typedef boost::unordered_map<int, float> ParamMomentum;

  struct DataTable {
    VectorClock vec_clock;
    VectorClock mirror_clock;
		VectorClock dcg_clock;
    iter_t travel_leave_clock;
    iter_t travel_arrive_clock;
    iter_t travel_apply_clock;
    iter_t global_clock;
    iter_t inter_dcg_clock;
		iter_t local_clock;
    Row2Index row2idx;
    size_t row_count;
    DataStorage store;
    DataStorage local_update;
    DataStorage update_history;
		DataStorage accu_update;
		DataStorage intra_dcg_accu_update;
		DataStorage inter_dcg_accu_update;
    DataStorage travel_model_store;
    ParamOffset param_offset_list;
#if (ENABLE_UPDATE_DIST)
		DataStorage shadow_accu_update[UPDATE_DIST_COL_NUM];
#endif
  };
  typedef std::vector<DataTable> DataTables;

	struct PendingMirrorUpdate {
    bool initted;
		iter_t data_age;
		uint table_id;
		vector<col_idx_t>* col_indexes;		
		vector<val_t>* col_updates;
		bool send_col;
		size_t send_size;
    PendingMirrorUpdate() : initted(false) {};
    void init(iter_t data_age_i, uint table_id_i, 
							vector<col_idx_t>* col_indexes_i, 
							vector<val_t>* col_updates_i,
							bool send_col_i, size_t send_size_i) {
			initted = true;
			data_age = data_age_i;
      table_id = table_id_i;
			col_indexes = col_indexes_i;
			col_updates = col_updates_i;
			send_col = send_col_i;
			send_size = send_size_i;
    }
  };
	typedef std::vector<PendingMirrorUpdate> PendingMirrorUpdates;

 private:
  uint channel_id;
  uint num_channels;
  uint process_id;
  uint num_processes;
  uint num_clients;

  boost::unordered_map<std::string, table_id_t> table_directory;

  shared_ptr<ServerClientEncode> communicator;
	shared_ptr<MirrorClientEncode> mc_communicator;

  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;

  DataTables data_tables;

  GeePsConfig config;

  tbb::tick_count begin_time;
  tbb::tick_count first_come_time;
	tbb::tick_count last_mirror_send_time;
	size_t last_mirror_send_size;
	tbb::tick_count top_layer_mirror_send_time;
	iter_t top_layer_send_clock;
	size_t iter_mirror_send_size;
	float update_threshold_adj;
	double wan_bandwidth_limit_byte_per_sec;
	PendingMirrorUpdates pending_mirror_updates;
  ParamLearningRate param_learning_rates;
  ParamMomentum param_momentums;

	boost::shared_ptr<boost::mutex> tablet_mutex;
	std::set<uint> aggr_table_group;

	bool is_dcg_router;

 private:
  template<class T>
  void resize_storage(vector<T>& storage, uint size) {
    if (storage.capacity() <= size) {
      uint capacity = get_nearest_power2(size);
      storage.reserve(capacity);
      // cerr << "capacity is " << capacity << endl;
    }
    if (storage.size() <= size) {
      storage.resize(size);
      // cerr << "size is " << size << endl;
    }
  }
  void process_multiclient_pending_reads(
      iter_t clock, uint table_id);
  void process_pending_reads(
      uint client_id, iter_t clock, uint table_id);
  void apply_updates(
      uint table_id, RowOpVal *update_rows, size_t batch_size, iter_t clock);
  void apply_local_updates_to_store(uint table_id, iter_t clock);
  void apply_gaia_updates(uint table_id, val_t *update, size_t num_vals, iter_t clock);
  void apply_dgc_updates(uint table_id, val_t *update, size_t num_vals, iter_t clock);
  void apply_fed_avg_updates(uint table_id, val_t *update, size_t num_vals, iter_t clock);
  void apply_fed_avg_weights(uint table_id);
  void update_fed_avg_stats(uint table_id, iter_t clock);
  bool send_mirror_updates(
      uint table_id, iter_t clock); 
	bool check_and_send_mirror_update(iter_t data_age, uint table_id, 
																		vector<col_idx_t>* col_indexes, 
																		vector<val_t>* col_updates);

  void send_dcg_mirror_updates(
      uint table_id, iter_t clock, bool inter_dcg);

  void send_dgc_mirror_updates(uint table_id, iter_t clock); 

  void check_and_apply_travel_model(uint table_id, bool kick_clock);


#if (ENABLE_UPDATE_DIST)
  void calc_update_dist(uint table_id, iter_t clock);
#endif

	uint get_dc_id(uint machine_id);
	uint get_dcg_id(uint machine_id);
	iter_t local_clock_min(vector<iter_t> clocks);
	iter_t intra_dcg_clock_min(iter_t local_clock, vector<iter_t> mirror_clocks);
  iter_t travel_leave_clock_min(iter_t local_clock, 
                                vector<iter_t> mirror_clocks, 
                                uint table_id);	
	iter_t global_clock_min(iter_t local_clock, 
													vector<iter_t> mirror_clocks, 
													vector<iter_t> dcg_clocks,
                          iter_t travel_apply_clock,
                          iter_t old_global_clock,
													uint table_id);	
  val_t *get_dcg_accu_buffer(uint table_id, uint from_server_id);

 public:
  TabletStorage(
      uint channel_id, uint num_channels, uint process_id, uint num_processes,
      shared_ptr<ServerClientEncode> communicator,
			shared_ptr<MirrorClientEncode> mc_communicator,
      cudaStream_t cuda_stream, cublasHandle_t cublas_handle,
      const GeePsConfig& config);
  void update_row_batch(
      uint client_id, iter_t clock, uint table_id,
      RowKey *row_keys, RowOpVal *updates, uint batch_size);
	void apply_mirror_updates(
      uint table_id, val_t *update_vals, size_t num_vals, iter_t clock,
			uint from_server_id, uint relay_dcg_update);
	void apply_mirror_updates(
      uint table_id, col_idx_t* col_indexes, 
			val_t* col_updates, size_t col_num, iter_t clock, 
			uint from_server_id, uint relay_dcg_update);
  void recv_model_travel(
      uint table_id, val_t *vals, size_t num_vals, iter_t clock,
			uint from_server_id);
  void recv_model_value(
      uint table_id, val_t *vals, size_t num_vals, iter_t clock,
			uint from_server_id);
  void clock(uint client_id, iter_t clock, uint table_id, bool no_update);
	void mirror_clock(uint server_id, iter_t clock, uint table_id);
	void inter_dcg_clock(uint server_id, iter_t clock, uint table_id);
  void get_stats(
      uint client_id, shared_ptr<MetadataServer> metadata_server);
    /* Now it also needs the stats from the MetadataServer.
     * This is just a work-around, and we need to fix it in the future.
     */
  void param_table_offset(const std::vector<ParamTableRowOffset>& param_offset_info);
  void param_learning_rate(int global_param_id, float learning_rate);
  void param_momentum(int global_param_id, float momentum);
  void reset_perf_counters();
};


void server(uint channel_id, uint nr_channels, uint server_id, uint nr_servers,
            boost::shared_ptr<zmq::context_t> zmq_ctx,
            const GeePsConfig& config);

#endif  // defined __tablet_server_hpp__
