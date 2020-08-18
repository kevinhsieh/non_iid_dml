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
#include "mirror-server-encoder-decoder.hpp"
#include "tablet-server.hpp"
#include "metadata-server.hpp"
#include "common/internal-config.hpp"

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;

void MirrorClientEncode::send_mirror_update(iter_t data_age, uint table_id, 
																						vector<col_idx_t>& col_indexes, 
																						vector<val_t>& col_updates,
																						RowOpVal *update_rows, size_t batch_size,
																						bool send_col, 
																						MirrorDest mirror_dest,
																						bool relay_dcg_update)
{
	vector<ZmqPortableBytes> msgs;
  msgs.resize(3);

	msgs[0].init_size(sizeof(mcs_mirror_update_batch_msg_t));
	mcs_mirror_update_batch_msg_t *mcs_mirror_update_batch_msg =
    reinterpret_cast<mcs_mirror_update_batch_msg_t *>(msgs[0].data());

	mcs_mirror_update_batch_msg->cmd = MIRROR_UPDATE_BATCH;
	mcs_mirror_update_batch_msg->server_id = server_id;
	mcs_mirror_update_batch_msg->data_age = data_age;
  mcs_mirror_update_batch_msg->table_id = table_id;
	mcs_mirror_update_batch_msg->relay_dcg_update = relay_dcg_update;

	/* Determine whether we should send selective columns or all columns */
	if (send_col)
	{
		msgs[1].pack_memory(col_indexes.data(), sizeof(col_idx_t) * col_indexes.size());
		msgs[2].pack_memory(col_updates.data(), sizeof(val_t) * col_updates.size());
	} else {
		msgs[1].init_size(sizeof(col_idx_t));
		col_idx_t *col = reinterpret_cast<col_idx_t *>(msgs[1].data());
		*col = UPDATE_ALL_COLUMNS;
		msgs[2].pack_memory(update_rows, sizeof(RowOpVal) * batch_size);
	}

	switch(mirror_dest)
	{
	case MIRROR_DEST_ALL:
		CHECK(!config.enable_overlay_network);
		mc_router_handler->send_to(mserver_names, msgs);
		break;
	case MIRROR_DEST_INTRA_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(intra_dcg_mserver_names, msgs);
		break;
	case MIRROR_DEST_INTER_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(inter_dcg_mserver_names, msgs);
		break;
	}
	
	return;
}

void MirrorClientEncode::send_model_travel(iter_t data_age, uint table_id, 
                                           RowOpVal *rows, size_t batch_size,
                                           MirrorDest mirror_dest)
{
  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);

	msgs[0].init_size(sizeof(mcs_model_travel_msg_t));
	mcs_model_travel_msg_t *mcs_model_travel_msg =
    reinterpret_cast<mcs_model_travel_msg_t *>(msgs[0].data());

	mcs_model_travel_msg->cmd = MODEL_TRAVEL;
	mcs_model_travel_msg->server_id = server_id;
	mcs_model_travel_msg->data_age = data_age;
  mcs_model_travel_msg->table_id = table_id;

  msgs[1].pack_memory(rows, sizeof(RowOpVal) * batch_size);

	switch(mirror_dest)
	{
	case MIRROR_DEST_ALL:
		CHECK(!config.enable_overlay_network);
		mc_router_handler->send_to(mserver_names, msgs);
		break;
	case MIRROR_DEST_INTRA_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(intra_dcg_mserver_names, msgs);
		break;
	case MIRROR_DEST_INTER_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(inter_dcg_mserver_names, msgs);
		break;
	}
	
	return;
}


void MirrorClientEncode::send_model_value(iter_t data_age, uint table_id, 
                                          RowOpVal *rows, size_t batch_size,
                                          MirrorDest mirror_dest)
{
  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);

	msgs[0].init_size(sizeof(mcs_model_value_msg_t));
	mcs_model_value_msg_t *mcs_model_value_msg =
    reinterpret_cast<mcs_model_value_msg_t *>(msgs[0].data());

	mcs_model_value_msg->cmd = MODEL_VALUE;
	mcs_model_value_msg->server_id = server_id;
	mcs_model_value_msg->data_age = data_age;
  mcs_model_value_msg->table_id = table_id;

  msgs[1].pack_memory(rows, sizeof(RowOpVal) * batch_size);

	switch(mirror_dest)
	{
	case MIRROR_DEST_ALL:
		CHECK(!config.enable_overlay_network);
		mc_router_handler->send_to(mserver_names, msgs);
		break;
	case MIRROR_DEST_INTRA_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(intra_dcg_mserver_names, msgs);
		break;
	case MIRROR_DEST_INTER_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(inter_dcg_mserver_names, msgs);
		break;
	}
	
	return;
}

void MirrorClientEncode::send_mirror_clock(iter_t clock, uint table_id, 
																					 MirrorDest mirror_dest)
{
	vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(mcs_mirror_clock_msg_t));
  mcs_mirror_clock_msg_t *mcs_mirror_clock_msg =
    reinterpret_cast<mcs_mirror_clock_msg_t *>(msgs[0].data());
  mcs_mirror_clock_msg->cmd = MIRROR_CLOCK;
  mcs_mirror_clock_msg->server_id = server_id;
  mcs_mirror_clock_msg->clock = clock;
	mcs_mirror_clock_msg->table_id = table_id;

	switch(mirror_dest)
	{
	case MIRROR_DEST_ALL:
		CHECK(!config.enable_overlay_network);
		mc_router_handler->send_to(mserver_names, msgs);
		break;
	case MIRROR_DEST_INTRA_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(intra_dcg_mserver_names, msgs);
		break;
	case MIRROR_DEST_INTER_DCG:
		CHECK(config.enable_overlay_network);
		mc_router_handler->send_to(inter_dcg_mserver_names, msgs);
		break;
	}

	return;
}

/* Send inter-DCG clock to servers within the same DCG */
void MirrorClientEncode::send_inter_dcg_clock(iter_t clock, uint table_id)
{
	CHECK(config.enable_overlay_network);

	vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(mcs_inter_dcg_clock_msg_t));
  mcs_inter_dcg_clock_msg_t *mcs_inter_dcg_clock_msg =
    reinterpret_cast<mcs_inter_dcg_clock_msg_t *>(msgs[0].data());
  mcs_inter_dcg_clock_msg->cmd = INTER_DCG_CLOCK;
  mcs_inter_dcg_clock_msg->server_id = server_id;
  mcs_inter_dcg_clock_msg->clock = clock;
	mcs_inter_dcg_clock_msg->table_id = table_id;

	mc_router_handler->send_to(intra_dcg_mserver_names, msgs);

	return;
}

string MirrorClientEncode::get_mc_router_stats()
{
	return mc_router_handler->get_stats();
}

string MirrorClientEncode::get_ms_router_stats()
{
	return ms_router_handler->get_stats();
}

MirrorClientDecode::MirrorClientDecode(
	shared_ptr<TabletStorage> storage) :
	storage(storage)
{
}

void MirrorClientDecode::decode_msg(
    const string& src, vector<ZmqPortableBytes>& msgs) {
  CHECK_GE(msgs.size(), 1);
  CHECK_GE(msgs[0].size(), sizeof(command_t));
  command_t cmd;
  msgs[0].unpack<command_t>(cmd);
  switch (cmd) {
  default:
    CHECK(0)
        << "Mirror client received unknown command: " << static_cast<int>(cmd)
        << " size: " << msgs[0].size();
  }
}

void MirrorClientDecode::router_callback(const string& src,
    vector<ZmqPortableBytes>& msgs) {
  decode_msg(src, msgs);
}

RouterHandler::RecvCallback MirrorClientDecode::get_recv_callback() {
  return bind(&MirrorClientDecode::router_callback, this, _1, _2);
}


void MirrorServerDecode::mirror_update_batch(
	const string& src, vector<ZmqPortableBytes>& args)
{
	CHECK_GE(args.size(), 3);
  CHECK_EQ(args[0].size(), sizeof(mcs_mirror_update_batch_msg_t));

	mcs_mirror_update_batch_msg_t *mcs_mirror_update_batch_msg =
    reinterpret_cast<mcs_mirror_update_batch_msg_t *>(args[0].data());

	//uint32_t server_id = mcs_mirror_update_batch_msg->server_id;
	iter_t clock = mcs_mirror_update_batch_msg->data_age;
  uint32_t table_id = mcs_mirror_update_batch_msg->table_id;

	col_idx_t* col_indexes = 
		reinterpret_cast<col_idx_t *>(args[1].data());

	size_t update_col_num = args[1].size() / sizeof(col_idx_t);

	val_t* col_updates = 
		reinterpret_cast<val_t *>(args[2].data());

	if (update_col_num == 1 && col_indexes[0] == UPDATE_ALL_COLUMNS) {
		/* Update all columns */
		storage->apply_mirror_updates(table_id, col_updates, 
																	args[2].size() / sizeof(val_t), clock,
																	mcs_mirror_update_batch_msg->server_id,
																	mcs_mirror_update_batch_msg->relay_dcg_update);
	} else {
		/* Update selective columns */		
		CHECK_EQ(update_col_num, args[2].size() / sizeof(val_t));
		storage->apply_mirror_updates(table_id, col_indexes, col_updates, 
																	update_col_num, clock,
																	mcs_mirror_update_batch_msg->server_id,
																	mcs_mirror_update_batch_msg->relay_dcg_update);
	}

	for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void MirrorServerDecode::mirror_clock(
	const string& src, vector<ZmqPortableBytes>& args)
{
	CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(mcs_mirror_clock_msg_t));
  mcs_mirror_clock_msg_t *mcs_mirror_clock_msg =
      reinterpret_cast<mcs_mirror_clock_msg_t *>(args[0].data());
  uint server_id = mcs_mirror_clock_msg->server_id;
  iter_t clock = mcs_mirror_clock_msg->clock;
  uint table_id = mcs_mirror_clock_msg->table_id;

  storage->mirror_clock(server_id, clock, table_id);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void MirrorServerDecode::inter_dcg_clock(
	const string& src, vector<ZmqPortableBytes>& args)
{
	CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(mcs_inter_dcg_clock_msg_t));
  mcs_inter_dcg_clock_msg_t *mcs_inter_dcg_clock_msg =
      reinterpret_cast<mcs_inter_dcg_clock_msg_t *>(args[0].data());
  uint server_id = mcs_inter_dcg_clock_msg->server_id;
  iter_t clock = mcs_inter_dcg_clock_msg->clock;
  uint table_id = mcs_inter_dcg_clock_msg->table_id;

  storage->inter_dcg_clock(server_id, clock, table_id);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void MirrorServerDecode::model_travel(
	const string& src, vector<ZmqPortableBytes>& args)
{
	CHECK_GE(args.size(), 2);
  CHECK_EQ(args[0].size(), sizeof(mcs_model_travel_msg_t));

	mcs_model_travel_msg_t *mcs_model_travel_msg =
    reinterpret_cast<mcs_model_travel_msg_t *>(args[0].data());

	//uint32_t server_id = mcs_model_travel_msg->server_id;
	iter_t clock = mcs_model_travel_msg->data_age;
  uint32_t table_id = mcs_model_travel_msg->table_id;

  storage->recv_model_travel(table_id,
                             reinterpret_cast<val_t *>(args[1].data()),
                             args[1].size() / sizeof(val_t), 
                             clock,
                             mcs_model_travel_msg->server_id);

	for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void MirrorServerDecode::model_value(
	const string& src, vector<ZmqPortableBytes>& args)
{
	CHECK_GE(args.size(), 2);
  CHECK_EQ(args[0].size(), sizeof(mcs_model_value_msg_t));

	mcs_model_value_msg_t *mcs_model_value_msg =
    reinterpret_cast<mcs_model_value_msg_t *>(args[0].data());

	//uint32_t server_id = mcs_model_value_msg->server_id;
	iter_t clock = mcs_model_value_msg->data_age;
  uint32_t table_id = mcs_model_value_msg->table_id;

  storage->recv_model_value(table_id,
                            reinterpret_cast<val_t *>(args[1].data()),
                            args[1].size() / sizeof(val_t), 
                            clock,
                            mcs_model_value_msg->server_id);

	for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}



void MirrorServerDecode::decode_msg(
    const string& src, vector<ZmqPortableBytes>& msgs) {
  CHECK_GE(msgs.size(), 1);
  CHECK_GE(msgs[0].size(), sizeof(command_t));
  command_t cmd;
  msgs[0].unpack<command_t>(cmd);
  switch (cmd) 
	{
	case MIRROR_UPDATE_BATCH:
		mirror_update_batch(src, msgs);
		break;
		
	case MIRROR_CLOCK:
		mirror_clock(src, msgs);
		break;

  case INTER_DCG_CLOCK:
		inter_dcg_clock(src, msgs);
		break;

  case MODEL_TRAVEL:
    model_travel(src, msgs);
    break;

  case MODEL_VALUE:
    model_value(src, msgs);
    break;

  default:
    CHECK(0)
			<< "Mirror server: " << server_id 
			<< "received unknown command: " << static_cast<int>(cmd)			
			<< " size: " << msgs[0].size()
			<< " from: " << src;
  }
}

void MirrorServerDecode::router_callback(const string& src,
    vector<ZmqPortableBytes>& msgs) 
{
  decode_msg(src, msgs);
}

RouterHandler::RecvCallback MirrorServerDecode::get_recv_callback() 
{
  return bind(&MirrorServerDecode::router_callback, this, _1, _2);
}

