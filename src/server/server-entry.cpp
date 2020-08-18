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

#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <string>
#include <vector>

#include "server-entry.hpp"
#include "server-encoder-decoder.hpp"
#include "mirror-server-encoder-decoder.hpp"
#include "tablet-server.hpp"
#include "metadata-server.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using boost::format;
using boost::shared_ptr;
using boost::make_shared;

void ServerThreadEntry::server_entry(
    uint channel_id, uint num_channels,
    uint process_id, uint num_processes,
    shared_ptr<zmq::context_t> zmq_ctx,
    const GeePsConfig& config) {
  uint port = config.tcp_base_port + channel_id;
	uint mport = config.tcp_base_port + num_channels + channel_id;
  string request_url = "tcp://*:" + boost::lexical_cast<std::string>(port);
	string m_request_url = "tcp://*:" + boost::lexical_cast<std::string>(mport);

  /* Basic checks */
  CHECK_LE(config.max_iter, UPDATE_DIST_ITER_NUM * UPDATE_DIST_ITER_INTERVAL);

  /* Init cuda stream and cublas handle */
  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  CUDA_CHECK(cudaStreamCreate(&cuda_stream));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, cuda_stream));

  /* Init communication */
  vector<string> connect_list;   /* Empty connect to */
  vector<string> bind_list;
  bind_list.push_back(request_url);
  string tablet_name = (format("tablet-%i") % process_id).str();
  shared_ptr<RouterHandler> router_handler = make_shared<RouterHandler>(
      channel_id, zmq_ctx, connect_list, bind_list, tablet_name,
      config, false);

  shared_ptr<ServerClientEncode> encoder = make_shared<ServerClientEncode>(
      router_handler, cuda_stream, cublas_handle,
      num_processes, process_id, config);

  shared_ptr<MirrorClientEncode> mc_encoder;
  shared_ptr<RouterHandler> ms_router_handler;
  shared_ptr<RouterHandler> mc_router_handler;

  if (config.enable_decentral)
  {
    // Create router for mirror server connection
    vector<string> connect_mc;		
    vector<string> bind_mc;			 /* Empty bind list */
    /* Build the connect list to the mirror tables in other clusters */
    uint tablet_group_base = process_id - (process_id % config.num_dc);
    for (int i = 0; i < config.num_dc; i ++) {
      if ((tablet_group_base + i) != process_id) {
        string connect_endpoint =
          "tcp://" + config.host_list[tablet_group_base + i] 
          + ":" + boost::lexical_cast<std::string>(mport);
        connect_mc.push_back(connect_endpoint);
      }
    }

    string tablet_mc_name = (format("mc-tablet-%i") % process_id).str();

    mc_router_handler = make_shared<RouterHandler>(
      channel_id, zmq_ctx, connect_mc, bind_mc, tablet_mc_name, 
      config, true);

    vector<string> connect_ms;	 /* Empty connect to */
    vector<string> bind_ms;
    bind_ms.push_back(m_request_url);
    string tablet_ms_name = (format("ms-tablet-%i") % process_id).str();
    ms_router_handler = make_shared<RouterHandler>(
      channel_id, zmq_ctx, connect_ms, bind_ms, tablet_ms_name, 
      config, true);

    mc_encoder = make_shared<MirrorClientEncode>(
      router_handler, mc_router_handler, ms_router_handler, 
			cuda_stream, cublas_handle,
      num_processes, process_id, config);
	}
  
  shared_ptr<TabletStorage> storage = make_shared<TabletStorage>(
      channel_id, num_channels, process_id, num_processes,
      encoder, mc_encoder, cuda_stream, cublas_handle, config);


  shared_ptr<MetadataServer> metadata_server = make_shared<MetadataServer>(
      channel_id, num_channels, process_id, num_processes,
      encoder, config);
  ClientServerDecode decoder(storage, metadata_server);

  if (config.enable_decentral)
  {
    MirrorClientDecode mc_decoder(storage);
	
    MirrorServerDecode ms_decoder(storage, process_id, config);

    mc_router_handler->start_handler_thread(mc_decoder.get_recv_callback());
    ms_router_handler->start_handler_thread(ms_decoder.get_recv_callback());
  }

  router_handler->do_handler(decoder.get_recv_callback());
}
