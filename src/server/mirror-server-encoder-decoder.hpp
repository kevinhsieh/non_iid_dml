#ifndef __mirror_server_encoder_decoder_hpp__
#define __mirror_server_encoder_decoder_hpp__

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

#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <map>

#include <boost/shared_ptr.hpp>

#include "common/wire-protocol.hpp"
#include "common/router-handler.hpp"

using boost::shared_ptr;
using boost::format;

class TabletStorage;
class MetadataServer;

enum MirrorDest {
	MIRROR_DEST_ALL,
	MIRROR_DEST_INTRA_DCG,
	MIRROR_DEST_INTER_DCG
};

/* Encodes messages to mirror server */
class MirrorClientEncode {
  shared_ptr<RouterHandler> router_handler;
	boost::shared_ptr<RouterHandler> mc_router_handler;
	boost::shared_ptr<RouterHandler> ms_router_handler;
	vector<string> mserver_names;
	vector<string> intra_dcg_mserver_names;
	vector<string> inter_dcg_mserver_names;
	uint server_id;
  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
	GeePsConfig config;

 public:
  explicit MirrorClientEncode(
      shared_ptr<RouterHandler> rh_i,
			shared_ptr<RouterHandler> rh_mc_i,
			shared_ptr<RouterHandler> rh_ms_i,
      cudaStream_t cuda_stream, cublasHandle_t cublas_handle,
      uint num_machines, uint server_id_i, const GeePsConfig& config_i)
        : router_handler(rh_i), mc_router_handler(rh_mc_i),
					ms_router_handler(rh_ms_i),
					server_id(server_id_i),
          cuda_stream(cuda_stream), cublas_handle(cublas_handle),
		      config(config_i) 
		{
			uint tablet_group_base = server_id - (server_id % config.num_dc);
			uint dc_id = (server_id % config.num_dc);
			for (int i = 0; i < config.num_dc; i++) {
        if ((tablet_group_base + i) != server_id) {
          string sname = (format("ms-tablet-%i") % (tablet_group_base + i)).str();
          mserver_names.push_back(sname);
					if (config.enable_overlay_network)
					{
						if (config.enable_olnw_multiple_routers)
						{
							if (config.dcg_router_list[dc_id] && 
									i == config.dcg_peer_router_list[dc_id]) 
							{
								inter_dcg_mserver_names.push_back(sname);
							}
						}
						else 
						{
							if (config.dcg_router_list[dc_id] && config.dcg_router_list[i]) 
							{
								inter_dcg_mserver_names.push_back(sname);
							}
						}

						if (config.dcg_id_list[dc_id] == config.dcg_id_list[i])
						{
							intra_dcg_mserver_names.push_back(sname);
						}
					}
        }
      }			
		}

	void send_mirror_update(iter_t data_age, uint table_id, 
													vector<col_idx_t>& col_indexes, 
													vector<val_t>& col_updates,
													RowOpVal *update_rows, size_t batch_size,
													bool send_col, MirrorDest mirror_dest,
													bool relay_dcg_update);
  void send_model_travel(iter_t data_age, uint table_id, 
                         RowOpVal *rows, size_t batch_size,
                         MirrorDest mirror_dest);
  void send_model_value(iter_t data_age, uint table_id, 
                        RowOpVal *rows, size_t batch_size,
                        MirrorDest mirror_dest);
	void send_mirror_clock(iter_t clock, uint table_id,
												 MirrorDest mirror_dest);
  void send_inter_dcg_clock(iter_t clock, uint table_id);
  string get_mc_router_stats();
	string get_ms_router_stats();
};

/* Decodes messages from a mirror server */
class MirrorClientDecode {
  shared_ptr<TabletStorage> storage;

 public:
  explicit MirrorClientDecode(
      shared_ptr<TabletStorage> storage);
  void decode_msg(const string& src, vector<ZmqPortableBytes>& msgs);
  void router_callback(const string& src, vector<ZmqPortableBytes>& msgs);

  RouterHandler::RecvCallback get_recv_callback();
};


/* Decodes messages from a mirror client */
class MirrorServerDecode {
  shared_ptr<TabletStorage> storage;
	uint server_id;
	GeePsConfig config;

	void mirror_update_batch(const string& src, vector<ZmqPortableBytes>& args);
  void mirror_clock(const string& src, vector<ZmqPortableBytes>& args);
	void inter_dcg_clock(const string& src, vector<ZmqPortableBytes>& args);
  void model_travel(const string& src, vector<ZmqPortableBytes>& args);
  void model_value(const string& src, vector<ZmqPortableBytes>& args);

 public:
  explicit MirrorServerDecode(
      shared_ptr<TabletStorage> storage, uint server_id, 
			const GeePsConfig& config)
		: storage(storage), server_id(server_id), config(config)
		{			
		}
  void decode_msg(const string& src, vector<ZmqPortableBytes>& msgs);
  void router_callback(const string& src, vector<ZmqPortableBytes>& msgs);

  RouterHandler::RecvCallback get_recv_callback();
};

#endif  // defined __mirror_server_encoder_decoder_hpp__
