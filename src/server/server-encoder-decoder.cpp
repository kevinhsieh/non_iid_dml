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

#include "common/portable-bytes.hpp"
#include "server-encoder-decoder.hpp"
#include "tablet-server.hpp"
#include "metadata-server.hpp"
#include "common/internal-config.hpp"

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;


ClientServerDecode::ClientServerDecode(
    shared_ptr<TabletStorage> storage,
    shared_ptr<MetadataServer> metadata_server) :
      storage(storage), metadata_server(metadata_server) {
}

void ClientServerDecode::find_row(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(cs_find_row_msg_t));
  cs_find_row_msg_t *cs_find_row_msg =
      reinterpret_cast<cs_find_row_msg_t *>(args[0].data());
  uint client_id = cs_find_row_msg->client_id;
  table_id_t table = cs_find_row_msg->table;
  row_idx_t row = cs_find_row_msg->row;

  metadata_server->find_row(src, client_id, table, row);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::clock(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(cs_clock_msg_t));
  cs_clock_msg_t *cs_clock_msg =
      reinterpret_cast<cs_clock_msg_t *>(args[0].data());
  uint client_id = cs_clock_msg->client_id;
  iter_t clock = cs_clock_msg->clock;
  uint table_id = cs_clock_msg->table_id;

  storage->clock(client_id, clock, table_id, false);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::clock_with_updates_batch(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_GE(args.size(), 3);
  CHECK_EQ(args[0].size(), sizeof(cs_clock_with_updates_batch_msg_t));
  cs_clock_with_updates_batch_msg_t *cs_clock_with_updates_batch_msg =
    reinterpret_cast<cs_clock_with_updates_batch_msg_t *>(args[0].data());
  uint client_id = cs_clock_with_updates_batch_msg->client_id;
  iter_t clock = cs_clock_with_updates_batch_msg->clock;
  uint table_id = cs_clock_with_updates_batch_msg->table_id;

  RowKey *row_keys =
    reinterpret_cast<RowKey *>(args[1].data());
  uint batch_size = args[1].size() / sizeof(RowKey);
  RowOpVal *updates =
    reinterpret_cast<RowOpVal *>(args[2].data());
  CHECK_EQ(batch_size, args[2].size() / sizeof(RowOpVal));

  tbb::tick_count inc_start = tbb::tick_count::now();
  storage->update_row_batch(
      client_id, clock, table_id, row_keys, updates, batch_size);
  storage->server_stats.inc_time +=
      (tbb::tick_count::now() - inc_start).seconds();
  storage->clock(client_id, clock, table_id, false);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::read_row_batch(
      const string& src, vector<ZmqPortableBytes>& args) {
  CHECK(0);
}

void ClientServerDecode::add_access_info(
    const std::string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 2);
  CHECK_EQ(args[0].size(), sizeof(cs_add_access_info_msg_t));
  cs_add_access_info_msg_t *cs_add_access_info_msg =
    reinterpret_cast<cs_add_access_info_msg_t *>(args[0].data());
  uint client_id = cs_add_access_info_msg->client_id;

  vector<RowAccessInfo> access_info;
  args[1].unpack_vector<RowAccessInfo>(access_info);

  metadata_server->add_access_info(src, client_id, access_info);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::kick_clock(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(ss_kick_clock_msg_t));
  ss_kick_clock_msg_t *ss_kick_clock_msg =
      reinterpret_cast<ss_kick_clock_msg_t *>(args[0].data());
  uint server_id = ss_kick_clock_msg->server_id;
  uint table_id = ss_kick_clock_msg->table_id;

  storage->clock(server_id, INITIAL_DATA_AGE, table_id, true);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}


void ClientServerDecode::get_stats(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(cs_get_stats_msg_t));
  cs_get_stats_msg_t *cs_get_stats_msg =
    reinterpret_cast<cs_get_stats_msg_t *>(args[0].data());
  uint client_id = cs_get_stats_msg->client_id;

  storage->get_stats(client_id, metadata_server);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::param_table_offset(
  const string& src, vector<ZmqPortableBytes>& args) {

  CHECK_EQ(args.size(), 2);
  CHECK_EQ(args[0].size(), sizeof(cs_param_table_offset_msg_t));

  std::vector<ParamTableRowOffset> param_offset_info;
  args[1].unpack_vector<ParamTableRowOffset>(param_offset_info);

  storage->param_table_offset(param_offset_info);
  
  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::param_learning_rate(
  const string& src, vector<ZmqPortableBytes>& args) {
  
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(cs_param_learning_rate_msg_t));

  cs_param_learning_rate_msg_t *cs_param_learning_rate_msg =
    reinterpret_cast<cs_param_learning_rate_msg_t *>(args[0].data());

  storage->param_learning_rate(
    cs_param_learning_rate_msg->param_id, 
    cs_param_learning_rate_msg->learning_rate);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::param_momentum(
  const string& src, vector<ZmqPortableBytes>& args) {
  
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(cs_param_momentum_msg_t));

  cs_param_momentum_msg_t *cs_param_momentum_msg =
    reinterpret_cast<cs_param_momentum_msg_t *>(args[0].data());

  storage->param_momentum(
    cs_param_momentum_msg->param_id, 
    cs_param_momentum_msg->momentum);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientServerDecode::decode_msg(
    const string& src, vector<ZmqPortableBytes>& msgs) {
  CHECK_GE(msgs.size(), 1);
  CHECK_GE(msgs[0].size(), sizeof(command_t));
  command_t cmd;
  msgs[0].unpack<command_t>(cmd);
  switch (cmd) {
  case FIND_ROW:
    find_row(src, msgs);
    break;
  case CLOCK_WITH_UPDATES_BATCH:
    clock_with_updates_batch(src, msgs);
    break;
  case READ_ROW_BATCH:
    read_row_batch(src, msgs);
    break;
  case CLOCK:
    clock(src, msgs);
    break;
  case ADD_ACCESS_INFO:
    add_access_info(src, msgs);
    break;
  case GET_STATS:
    get_stats(src, msgs);
    break;
  case KICK_CLOCK:
    kick_clock(src, msgs);
    break;
  case SEND_PARAM_TABLE_OFFSET:
    param_table_offset(src, msgs);
    break;
  case SET_PARAM_LEARNING_RATE:
    param_learning_rate(src, msgs);
    break;
  case SET_PARAM_MOMENTUM:
    param_momentum(src, msgs);
    break;
  default:
    CHECK(0)
        << "Server received unknown command: " << static_cast<int>(cmd)
        << " size: " << msgs[0].size();
  }
}

void ClientServerDecode::router_callback(const string& src,
    vector<ZmqPortableBytes>& msgs) {
  decode_msg(src, msgs);
}

RouterHandler::RecvCallback ClientServerDecode::get_recv_callback() {
  return bind(&ClientServerDecode::router_callback, this, _1, _2);
}


void ServerClientEncode::clock(uint server_id, iter_t clock) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(sc_clock_msg_t));
  sc_clock_msg_t *sc_clock_msg =
    reinterpret_cast<sc_clock_msg_t *>(msgs[0].data());
  sc_clock_msg->cmd = CLOCK;
  sc_clock_msg->server_id = server_id;
  sc_clock_msg->clock = clock;

  /* Broadcast to all clients */
  router_handler->send_to(client_names, msgs);
}


void ServerClientEncode::send_local_clock_kick(uint table_id) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(ss_kick_clock_msg_t));
  ss_kick_clock_msg_t *ss_kick_clock_msg =
    reinterpret_cast<ss_kick_clock_msg_t *>(msgs[0].data());
  ss_kick_clock_msg->cmd = KICK_CLOCK;
  ss_kick_clock_msg->server_id = server_id;
  ss_kick_clock_msg->table_id = table_id;

  /* Send to itself */
  router_handler->send_to_self(msgs);
}


void ServerClientEncode::find_row(
      uint client_id, table_id_t table, row_idx_t row, uint32_t server_id) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(sc_find_row_msg_t));
  sc_find_row_msg_t *sc_find_row_msg =
    reinterpret_cast<sc_find_row_msg_t *>(msgs[0].data());
  sc_find_row_msg->cmd = FIND_ROW;
  sc_find_row_msg->table = table;
  sc_find_row_msg->row = row;
  sc_find_row_msg->server_id = server_id;

  CHECK_LT(client_id, client_names.size());
  string client_name = client_names[client_id];
  router_handler->send_to(client_name, msgs);
}

void ServerClientEncode::read_row_batch_reply(
      uint client_id, uint server_id, iter_t data_age, iter_t self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(3);

  msgs[0].init_size(sizeof(sc_read_row_batch_msg_t));
  sc_read_row_batch_msg_t *sc_read_row_batch_msg =
    reinterpret_cast<sc_read_row_batch_msg_t *>(msgs[0].data());
  sc_read_row_batch_msg->cmd = READ_ROW_BATCH;
  sc_read_row_batch_msg->server_id = server_id;
  sc_read_row_batch_msg->data_age = data_age;
  sc_read_row_batch_msg->self_clock = self_clock;
  sc_read_row_batch_msg->table_id = table_id;

  msgs[1].pack_memory(row_keys, sizeof(RowKey) * batch_size);
  // msgs[2].pack_gpu_memory(row_data, sizeof(RowData) * batch_size, cuda_stream);
  msgs[2].pack_memory(row_data, sizeof(RowData) * batch_size);

  CHECK_LT(client_id, client_names.size());
  string client_name = client_names[client_id];
  router_handler->send_to(client_name, msgs);
}

void ServerClientEncode::get_stats(uint client_id, const string& stats) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);

  msgs[0].init_size(sizeof(sc_get_stats_msg_t));
  sc_get_stats_msg_t *sc_get_stats_msg =
    reinterpret_cast<sc_get_stats_msg_t *>(msgs[0].data());
  sc_get_stats_msg->cmd = GET_STATS;

  msgs[1].pack_string(stats);

  CHECK_LT(client_id, client_names.size());
  string client_name = client_names[client_id];
  router_handler->send_to(client_name, msgs);
}

string ServerClientEncode::get_router_stats() {
  return router_handler->get_stats();
}
