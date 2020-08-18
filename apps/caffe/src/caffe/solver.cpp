#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/format.hpp>

#include <tbb/tick_count.h>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::cout;
using std::cerr;
using std::endl;
using boost::make_shared;

#define LOCAL_DATA_IN_PS

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const PsConfig *ps_config)
    : ps_config_(*ps_config), net_(), callbacks_(), root_solver_(NULL),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  LOG(INFO) << "Solver scaffolding done.";
  iter_ = 0;
  current_step_ = 0;

  /* Initialize parameter server */
  InitPs();
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG(INFO) << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG(INFO) << "Creating training net from train_net file: "
              << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG(INFO) << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG(INFO) << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <>
void Solver<float>::PrepareAccessInfo() {
  vector<shared_ptr<Layer<float> > >& layers = this->net_->layers_;
  vector<string>& layer_types = this->net_->layer_types_;
  vector<string>& layer_names = this->net_->layer_names_;
  vector<bool>& layer_need_backward = this->net_->layer_need_backward_;
  vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward_;
  vector<shared_ptr<Blob<float> > >& params = this->net_->params_;
  bool print_ = this->ps_config_.debug ? ps_config_.worker_id == 0 : false;

  layer_infos_.resize(layers.size());
  int total_num_params = 0;
  int layer_count = 0;
  int table_id = 0;
  int row_id = 0;
  int local_store_row_id = 0;
  int global_param_id = 0;

  /* Decide row keys for model parameters */
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    shared_ptr<Layer<float> >& layer = layers[layer_id];
    LayerInfo& layer_info = layer_infos_[layer_id];
    int num_params = layer->blobs().size();
    if (print_) {
      LOG(INFO) << "Prepare Access for layer: " << layer_names[layer_id]
                << ", num params: " << num_params;
    }
    layer_info.layer_need_backward = layer_need_backward[layer_id];
    layer_info.bottom_need_backward = bottom_need_backward[layer_id];
    if (num_params > 0) {
      layer_info.param_infos.resize(num_params);
      total_num_params += num_params;
      if (layer_info.layer_need_backward) {
        layer_info.local_param = false;
      } else {
        /* The parameters of this layer will not be changed */
        layer_info.local_param = true;
      }
      layer_info.num_vals = 0;
      for (int param_id = 0; param_id < num_params; param_id++) {
        shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
        layer_info.param_infos[param_id].val_offset = layer_info.num_vals;
        layer_info.param_infos[param_id].global_param_id = global_param_id++;
        layer_info.num_vals += param->count();
      }
      int num_rows = (layer_info.num_vals + ROW_DATA_SIZE - 1) / ROW_DATA_SIZE;
      for (int i = 0; i < num_rows; i++) {
        if (!layer_info.local_param) {
          layer_info.row_ids.push_back(row_id++);
        } else {
          layer_info.row_ids.push_back(local_store_row_id++);
        }
        layer_info.history_data_row_ids.push_back(local_store_row_id++);
      }
      if (!layer_info.local_param) {
        layer_info.table_id = table_id;
        layer_count++;
        if (ps_config_.multi_table &&
            (layer_count % ps_config_.layers_per_table == 0)) {
          table_id++;
          row_id = 0;
        }
      } else {
        /* The parameters of this layer only need to be stored locally */
        layer_info.table_id = 0;
      }
    }
    layer_info.fw_read_time = 0;
    layer_info.fw_compute_time = 0;
    layer_info.fw_write_time = 0;
    layer_info.bw_read_time = 0;
    layer_info.bw_compute_time = 0;
    layer_info.bw_write_time = 0;
    layer_info.test_time = 0;
    layer_info.snapshot_model_time = 0;
    layer_info.snapshot_solverstate_time = 0;
    if (print_) {
      LOG(INFO) << "Param info size: " << layer_info.param_infos.size();
    }
  }
  CHECK_EQ(total_num_params, params.size());
  num_tables_ = row_id == 0 ? table_id : table_id + 1;
  

  /* Decide row keys for intermediate data blobs */
  vector<shared_ptr<Blob<float> > >& imbs = this->net_->blobs_;
  imb_data_infos_.resize(imbs.size());
  for (int imb_id = 0; imb_id < imbs.size(); imb_id++) {
    RowAccessInfo& imb_info = imb_data_infos_[imb_id];
    imb_info.num_vals = imbs[imb_id]->count();
    int num_rows = (imb_info.num_vals + ROW_DATA_SIZE - 1) / ROW_DATA_SIZE;
    for (int i = 0; i < num_rows; i++) {
      imb_info.row_ids.push_back(local_store_row_id++);
    }
    imb_info.data_in_mem = false;
    imb_info.data_handle = -1;
  }
  /* Decide row keys for intermediate diff blobs */
  imb_diff_infos_.resize(imbs.size());
  for (int imb_id = 0; imb_id < imbs.size(); imb_id++) {
    RowAccessInfo& imb_info = imb_diff_infos_[imb_id];
    imb_info.num_vals = imbs[imb_id]->count();
    int num_rows = (imb_info.num_vals + ROW_DATA_SIZE - 1) / ROW_DATA_SIZE;
    for (int i = 0; i < num_rows; i++) {
      imb_info.row_ids.push_back(local_store_row_id++);
    }
    imb_info.data_in_mem = false;
    imb_info.data_handle = -1;
  }

  /* Decide which intermediate blobs to access/release at each layer */
  vector<int>& net_output_blob_indices = this->net_->net_output_blob_indices_;
  IntSet net_output_set;
  for (int i = 0; i < net_output_blob_indices.size(); i++) {
    net_output_set[net_output_blob_indices[i]] = FetchKeep();
  }
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    vector<int>& bottom_imb_ids = this->net_->bottom_id_vecs_[layer_id];
    vector<int>& top_imb_ids = this->net_->top_id_vecs_[layer_id];
    IntSet& imbs_used_fw = layer_info.imbs_used_fw;
    IntSet& imb_diffs_used_fw = layer_info.imb_diffs_used_fw;
    IntSet& imbs_used_bw = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used_bw = layer_info.imb_diffs_used_bw;
		LOG(INFO) << "Layer " << layer_id << " maps to table "
							<< layer_info.table_id;
    for (int i = 0; i < bottom_imb_ids.size(); i++) {
      int blob_id = bottom_imb_ids[i];
      if (net_output_set.count(blob_id)) {
        LOG(INFO) << "Blob #" << blob_id << " is an output blob";
        /* Do not stream output blobs */
        continue;
      }
      /* In the forward pass, use (fetch, keep) all bottom data blobs */
      imbs_used_fw[blob_id] = FetchKeep(true, true);
      /* In the forward pass, use no bottom diff blobs */
      /* In the backward pass, use (fetch, no keep) all bottom data blobs,
       * except for Data layers */
      if (layer_types[layer_id] == "Data") {
        /* Not used */
      } else {
        imbs_used_bw[blob_id] = FetchKeep(true, false);
      }
      /* In the backward pass, use (no fetch, keep) all bottom diff blobs,
       * except for Data layers */
      if (layer_types[layer_id] == "Data") {
        /* Not used */
      } else {
        imb_diffs_used_bw[blob_id] = FetchKeep(false, true);
      }
    }
    for (int i = 0; i < top_imb_ids.size(); i++) {
      int blob_id = top_imb_ids[i];
      if (net_output_set.count(blob_id)) {
        /* Do not stream output blobs */
        LOG(INFO) << "Blob #" << blob_id << " is an output blob";
        continue;
      }
      /* In the forward pass, use (no fetch, keep) all top data blobs */
      imbs_used_fw[blob_id] = FetchKeep(false, true);
      /* In the forward pass, use (no fetch, keep) the top diff blobs
       * only in loss layers */
      if (layer_types[layer_id] == "SoftmaxWithLoss" ||
          layer_types[layer_id] == "CenterLoss") {
        imb_diffs_used_fw[blob_id] = FetchKeep(false, true);
      }
      /* In the backward pass, use (fetch, no keep) the top data blobs
       * only in ReLU, LRN, Pooling, BatchNorm, Dropout,
       * and SoftmaxWithLoss layers */
      if (layer_types[layer_id] == "ReLU" ||
          layer_types[layer_id] == "LRN" ||
          layer_types[layer_id] == "Pooling" ||
          layer_types[layer_id] == "BatchNorm" ||
          layer_types[layer_id] == "BatchReNorm" ||
          layer_types[layer_id] == "Dropout" ||
          layer_types[layer_id] == "SoftmaxWithLoss" ||
          layer_types[layer_id] == "CenterLoss") {
        imbs_used_bw[blob_id] = FetchKeep(true, false);
      }
      /* In the backward pass, use (fetch, no keep) all top diff blobs,
       * except for Data layers, top[1] of LRN, Pooling layers, BatchNorm,
       * and Dropout layers */
      if (layer_types[layer_id] == "Data" ||
          (layer_types[layer_id] == "LRN" && i > 0) ||
          (layer_types[layer_id] == "Pooling" && i > 0) ||
          (layer_types[layer_id] == "BatchNorm" && i > 0) ||
          (layer_types[layer_id] == "BatchReNorm" && i > 0) ||
          (layer_types[layer_id] == "Dropout" && i > 0)) {
        /* Do not use */
      } else {
        imb_diffs_used_bw[blob_id] = FetchKeep(true, false);
      }
    }
  }

  /* Decide imbs to accesss/release in forward pass */
  IntSet empty_set;
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    IntSet& imbs_used = layer_info.imbs_used_fw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_fw;
    /* Decide imbs to access in forward pass */
    vector<ImbInfo>& imbs_to_access = layer_info.imbs_to_access_fw;
    vector<ImbInfo>& imb_diffs_to_access = layer_info.imb_diffs_to_access_fw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_data_infos_[imb_id].data_in_mem) {
        imb_data_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imbs_to_access.push_back(imb_info);
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diff_infos_[imb_id].data_in_mem) {
        imb_diff_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imb_diffs_to_access.push_back(imb_info);
      }
    }
    /* Decide imbs to release in forward pass */
    vector<ImbInfo>& imbs_to_release = layer_info.imbs_to_release_fw;
    vector<ImbInfo>& imb_diffs_to_release = layer_info.imb_diffs_to_release_fw;
    /* Decide the next forward/backward layer */
    int next_layer_id = layer_id + 1;
    bool forward = true;
    if (next_layer_id >= layers.size()) {
      /* The next layer should be a backward layer */
      forward = false;
      next_layer_id = layers.size() - 1;
      while (next_layer_id >= 0) {
        if (layer_need_backward[next_layer_id]) {
          break;
        }
        next_layer_id--;
      }
    }
    IntSet *imbs_used_next_layer_ptr = NULL;
    IntSet *imb_diffs_used_next_layer_ptr = NULL;
    if (forward) {
      CHECK(next_layer_id >= 0 && next_layer_id < layers.size());
      imbs_used_next_layer_ptr =
          &layer_infos_[next_layer_id].imbs_used_fw;
      imb_diffs_used_next_layer_ptr =
          &layer_infos_[next_layer_id].imb_diffs_used_fw;
    } else {
      if (next_layer_id >= 0) {
        CHECK(next_layer_id < layers.size());
        CHECK(layer_need_backward[next_layer_id]);
        imbs_used_next_layer_ptr =
            &layer_infos_[next_layer_id].imbs_used_bw;
        imb_diffs_used_next_layer_ptr =
            &layer_infos_[next_layer_id].imb_diffs_used_bw;
      } else {
        /* The current layer is the last one, we should release all blobs */
        imbs_used_next_layer_ptr = &empty_set;
        imb_diffs_used_next_layer_ptr = &empty_set;
      }
    }
    /* Release the blobs that are not used in the next layer */
    IntSet& imbs_used_next_layer = *imbs_used_next_layer_ptr;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imbs_used_next_layer.count(imb_id)) {
        CHECK(imb_data_infos_[imb_id].data_in_mem);
        imb_data_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imbs_to_release.push_back(imb_info);
      }
    }
    IntSet& imb_diffs_used_next_layer = *imb_diffs_used_next_layer_ptr;
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diffs_used_next_layer.count(imb_id)) {
        CHECK(imb_diff_infos_[imb_id].data_in_mem);
        imb_diff_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imb_diffs_to_release.push_back(imb_info);
      }
    }
  }
  /* Decide imbs to accesss/release in backward pass */
  for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
    if (!layer_need_backward[layer_id]) {
      LOG(INFO) << "Layer " << layer_names[layer_id] << " doesn't need backward";
      continue;
    }
    LayerInfo& layer_info = layer_infos_[layer_id];
    IntSet& imbs_used = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_bw;
    vector<ImbInfo>& imbs_to_access = layer_info.imbs_to_access_bw;
    vector<ImbInfo>& imb_diffs_to_access = layer_info.imb_diffs_to_access_bw;
    /* Decide imbs to access in backward pass */
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_data_infos_[imb_id].data_in_mem) {
        imb_data_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imbs_to_access.push_back(imb_info);
      }
    }
    /* Decide imb diffs to access in backward pass */
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diff_infos_[imb_id].data_in_mem) {
        imb_diff_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imb_diffs_to_access.push_back(imb_info);
      }
    }
    /* Decide imbs to release in backward pass */
    vector<ImbInfo>& imbs_to_release = layer_info.imbs_to_release_bw;
    vector<ImbInfo>& imb_diffs_to_release = layer_info.imb_diffs_to_release_bw;
    int next_layer_id = layer_id - 1;
    while (next_layer_id >= 0) {
      if (layer_need_backward[next_layer_id]) {
        break;
      }
      next_layer_id--;
    }
    
    IntSet& imbs_used_next_layer = (next_layer_id >= 0) ?
        layer_infos_[next_layer_id].imbs_used_bw : empty_set;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imbs_used_next_layer.count(imb_id)) {
        CHECK(imb_data_infos_[imb_id].data_in_mem);
        imb_data_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imbs_to_release.push_back(imb_info);
      }
    }
    IntSet& imb_diffs_used_next_layer = (next_layer_id >= 0) ?
        layer_infos_[next_layer_id].imb_diffs_used_bw : empty_set;
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diffs_used_next_layer.count(imb_id)) {
        CHECK(imb_diff_infos_[imb_id].data_in_mem);
        imb_diff_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imb_diffs_to_release.push_back(imb_info);
      }
    }
  }
  /* All blobs should have been released */
  for (int i = 0; i < imb_data_infos_.size(); i++) {
    CHECK(!imb_data_infos_[i].data_in_mem) << "i = " << i;
  }
  for (int i = 0; i < imb_diff_infos_.size(); i++) {
    CHECK(!imb_diff_infos_[i].data_in_mem) << "i = " << i;
  }

  /* Prepare access handles */
  for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    layer_info.layer_handles.resize(ps_config_.batches_per_clock);
    for (int batch_id = 0; batch_id < ps_config_.batches_per_clock; batch_id++) {
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];
      layer_handles.imbs_to_access_fw.resize(layer_info.imbs_to_access_fw.size());
      layer_handles.imbs_to_release_fw.resize(layer_info.imbs_to_release_fw.size());
      layer_handles.imb_diffs_to_access_fw.resize(layer_info.imb_diffs_to_access_fw.size());
      layer_handles.imb_diffs_to_release_fw.resize(layer_info.imb_diffs_to_release_fw.size());
      layer_handles.imbs_to_access_bw.resize(layer_info.imbs_to_access_bw.size());
      layer_handles.imbs_to_release_bw.resize(layer_info.imbs_to_release_bw.size());
      layer_handles.imb_diffs_to_access_bw.resize(layer_info.imb_diffs_to_access_bw.size());
      layer_handles.imb_diffs_to_release_bw.resize(layer_info.imb_diffs_to_release_bw.size());
    }
  }
}

template <>
void Solver<float>::InitPs() {
  /* Decide rows to access at each layer */
  PrepareAccessInfo();
  vector<bool>& layer_need_backward = this->net_->layer_need_backward_;

  /* Initialize GeePS */
  ps_config_.geeps_config.num_tables = num_tables_;
  CHECK(ps_config_.geeps_config.host_list.size());
  ps_ = make_shared<GeePs>(ps_config_.worker_id, ps_config_.geeps_config);

  /* Virtual iteration */
  /* We hope the allocated space is contiguous in the cache, so:
   * on allocating, we first allocate model parameters,
   * and then allocate intermediate blobs (likely to be top blobs);
   * on releasing, we first release intermediate blobs
   * (likely to be bottom blobs), and then model parameters. */
  for (int batch_id = 0; batch_id < ps_config_.batches_per_clock; batch_id++) {
    /* Virtual iteration, forward pass */
    for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];
      /* Read model parameters */
      if (layer_info.param_infos.size()) {
        if (!layer_info.local_param) {
          layer_handles.read_handle = ps_->VirtualRead(
              layer_info.table_id, layer_info.row_ids, ps_config_.slack);
        } else {
          bool fetch = true;
          layer_handles.read_handle = ps_->VirtualLocalAccess(
              layer_info.row_ids, fetch);
        }
      }
#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_fw[i];
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imbs_to_access_fw.size());
        int& handle = layer_handles.imbs_to_access_fw[i];
        handle = ps_->VirtualLocalAccess(
            access_info.row_ids, imb_info.fetch);
        access_info.data_handle = handle;
      }
      /* Access intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_fw[i];
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_fw.size());
        int& handle = layer_handles.imb_diffs_to_access_fw[i];
        handle = ps_->VirtualLocalAccess(
            access_info.row_ids, imb_info.fetch);
        access_info.data_handle = handle;
      }
#endif
#if defined(LOCAL_DATA_IN_PS)
      /* Release intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_fw[i];
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imbs_to_release_fw.size());
        int& handle = layer_handles.imbs_to_release_fw[i];
        handle = ps_->VirtualPostLocalAccess(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
      }
      /* Release intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_fw[i];
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imb_diffs_to_release_fw.size());
        int& handle = layer_handles.imb_diffs_to_release_fw[i];
        handle = ps_->VirtualPostLocalAccess(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
      }
#endif
      /* Release model parameters */
      if (layer_info.param_infos.size()) {
        if (!layer_info.local_param) {
          layer_handles.postread_handle = ps_->VirtualPostRead(
              layer_handles.read_handle);
        } else {
          bool keep = false;  /* don't need to write back */
          layer_handles.postread_handle = ps_->VirtualPostLocalAccess(
              layer_handles.read_handle, keep);
        }
      }
    }
    /* Virtual iteration, backward pass */
    for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
      if (!layer_need_backward[layer_id]) {
        /* For a layer that doesn't need backward, we assume all layers
         * below it don't need backward either. */
        continue;
      }
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];
      /* Read and prewrite model parameters */
      if (layer_info.param_infos.size()) {
        CHECK(!layer_info.local_param);
        /* Copy param ids and value offsets to separate vectors */
        int num_params = layer_info.param_infos.size();
        vector<int> global_param_ids;
        vector<int> val_offsets;
        global_param_ids.resize(num_params);
        val_offsets.resize(num_params);
        for (int param_id = 0; param_id < layer_info.param_infos.size(); 
             param_id++) {
          global_param_ids[param_id] = layer_info.param_infos[param_id].global_param_id;
          val_offsets[param_id] = layer_info.param_infos[param_id].val_offset;
        }
        layer_handles.prewrite_handle = ps_->VirtualPreUpdate(
            layer_info.table_id, layer_info.row_ids, global_param_ids, val_offsets);
        layer_handles.bw_read_handle = ps_->VirtualRead(
            layer_info.table_id, layer_info.row_ids, ps_config_.slack);
        bool fetch = true;
        layer_handles.history_access_handle =
            ps_->VirtualLocalAccess(
                layer_info.history_data_row_ids, fetch);
      }
#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_data_infos_.size());
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imbs_to_access_bw.size());
        int& handle = layer_handles.imbs_to_access_bw[i];
        handle = ps_->VirtualLocalAccess(
            access_info.row_ids, imb_info.fetch);
        access_info.data_handle = handle;
      }
      /* Access intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_diff_infos_.size());
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_bw.size());
        int& handle = layer_handles.imb_diffs_to_access_bw[i];
        handle = ps_->VirtualLocalAccess(
            access_info.row_ids, imb_info.fetch);
        access_info.data_handle = handle;
      }
#endif
#if defined(LOCAL_DATA_IN_PS)
      /* Postaccess intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_data_infos_.size());
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imbs_to_release_bw.size());
        int& handle = layer_handles.imbs_to_release_bw[i];
        handle = ps_->VirtualPostLocalAccess(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
      }
      /* Postaccess intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_diff_infos_.size());
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imb_diffs_to_release_bw.size());
        int& handle = layer_handles.imb_diffs_to_release_bw[i];
        handle = ps_->VirtualPostLocalAccess(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
      }
#endif
      /* Postread and write model parameters */
      if (layer_info.param_infos.size()) {
        layer_handles.write_handle = ps_->VirtualUpdate(
            layer_handles.prewrite_handle);
        layer_handles.bw_postread_handle = ps_->VirtualPostRead(
            layer_handles.bw_read_handle);
        bool keep = true;
        layer_handles.history_postaccess_handle =
            ps_->VirtualPostLocalAccess(
                layer_handles.history_access_handle, keep);
      }
    }
  }
  ps_->VirtualClock();
  /* Report unrepeated accesses.
   * Currently, we assume all accesses after clock are unrepeated accesses. */
  for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    if (!layer_info.local_param) {
      continue;
    }
    /* Local parameters are only modified at initialization,
     * so we register the write accesses as unrepeated ones. */
    LayerHandles& layer_handles = layer_info.layer_handles[0];
    bool fetch = false;
    layer_handles.prewrite_handle = ps_->VirtualLocalAccess(
        layer_info.row_ids, fetch);
    bool keep = true;
    layer_handles.write_handle = ps_->VirtualPostLocalAccess(
        layer_handles.prewrite_handle, keep);
  }
  ps_->FinishVirtualIteration();
  LOG(INFO) << "Virtual iteration done";
}

template <>
void SGDSolver<float>::InitPsValues() {
  bool print_ = this->ps_config_.debug ? ps_config_.worker_id == 0 : false;

  vector<shared_ptr<Layer<float> > >& layers = this->net_->layers_;
  vector<bool>& layer_need_backward = this->net_->layer_need_backward_;
  vector<string>& layer_names = this->net_->layer_names_;
  /* Set initial parameter values */
  if (print_) {
    LOG(INFO) << "Set initial parameter values";
  }
  for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
    shared_ptr<Layer<float> >& layer = layers[layer_id];
    if (print_) {
      LOG(INFO) << "Layer " << layer_names[layer_id];
    }
    LayerInfo& layer_info = layer_infos_[layer_id];
    LayerHandles& layer_handles = layer_info.layer_handles[0];
    if (!layer_info.param_infos.size()) {
      continue;
    }
    if (layer_info.local_param || ps_config_.worker_id == 0) {
      /* Non-local parameters are stored in PS,
       * so only one worker needs to set it. */
      /* Pre-write */
      RowData *update_buffer = NULL;
      if (!layer_info.local_param) {
        ps_->PreUpdate(layer_handles.prewrite_handle, &update_buffer);
      } else {
        ps_->LocalAccess(layer_handles.prewrite_handle, &update_buffer);
      }
      float *params_vals = reinterpret_cast<float *>(update_buffer);
      for (int param_id = 0;
          param_id < layer_info.param_infos.size(); param_id++) {
        int param_val_offset = layer_info.param_infos[param_id].val_offset;
        float *param_vals = &params_vals[param_val_offset];
        shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
        bool change_head = false;
            /* "false" means that we will keep the head to be CPU_DATA.
             * We want to keep what's currently in CPU memory */
        param->set_gpu_data(param_vals, change_head);
        if (print_) {
          LOG(INFO) << "Set param pointer:" << param_vals;
        }
        /* "false" means that we don't change head here,
         * because we want to keep what's currently in CPU memory */
        /* Values are filled in CPU memory, do a gpu_data() call to copy them
         * to GPU memory */
        param->gpu_data();
        change_head = true;
            /* "true" means that we will change the head to UNINITIALIZED */
        param->set_gpu_data(NULL, change_head);
        CHECK_EQ(param->check_data_head(), SyncedMemory::UNINITIALIZED);
      }
      /* Write */
      if (!layer_info.local_param) {
        ps_->Update(layer_handles.write_handle);
      } else {
        ps_->PostLocalAccess(layer_handles.write_handle);
      }
    } else {
      CHECK(!layer_info.local_param && ps_config_.worker_id != 0);
      /* The other workers set their parameter data head to UNINITIALIZED */
      for (int param_id = 0;
          param_id < layer_info.param_infos.size(); param_id++) {
        shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
        bool change_head = true;
            /* "true" means that we will change the head to UNINITIALIZED */
        bool allow_reset_cpu_data = true;
        param->set_gpu_data(NULL, change_head, allow_reset_cpu_data);
      }
    }
  }
  if (print_) {
    LOG(INFO) << "Set initial updates history values";
  }
  for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    LayerHandles& layer_handles = layer_info.layer_handles[0];
    if (print_) {
      LOG(INFO) << "Layer " << layer_names[layer_id];
    }
    if (!layer_info.param_infos.size() || !layer_need_backward[layer_id]) {
      continue;
    }
    /* Pre-write */
    RowData *history_buffer = NULL;
    ps_->LocalAccess(layer_handles.history_access_handle, &history_buffer);
    float *history_vals = reinterpret_cast<float *>(history_buffer);
    for (int param_id = 0;
        param_id < layer_info.param_infos.size(); param_id++) {
      int param_val_offset = layer_info.param_infos[param_id].val_offset;
      float *history_param_vals = &history_vals[param_val_offset];
      int global_param_id =
          layer_info.param_infos[param_id].global_param_id;
      shared_ptr<Blob<float> >& updates_history = history_[global_param_id];
      CHECK(!updates_history->check_gpu_data());
      bool change_head = false;
          /* "false" means that we will keep the head to be CPU_DATA.
           * We want to keep what's currently in CPU memory */
      updates_history->set_gpu_data(history_param_vals, change_head);
      if (print_) {
        LOG(INFO) << "Set update history pointer:" << history_param_vals 
                  << ", Global Param ID:" << global_param_id;
      }
      /* Values are filled in CPU memory, do a gpu_data() call to copy them
       * to GPU memory */
      updates_history->gpu_data();
      change_head = true;
          /* "true" means that we will change the head to UNINITIALIZED */
      updates_history->set_gpu_data(NULL, change_head);
    }
    /* Write */
    CHECK_GT(layer_handles.history_postaccess_handle, 0);
    ps_->PostLocalAccess(layer_handles.history_postaccess_handle);
  }
  LOG(INFO) << "Set initial parameter values done";
  ps_->Clock();
  ps_->StartIterations();
  LOG(INFO) << "Iterations started";
}

template <>
float SGDSolver<float>::ForwardBackwardUsingPs(
    const vector<Blob<float>* >& bottom,
    const shared_ptr<Net<float> >& net,
    bool test, bool do_snapshot) {
  vector<shared_ptr<Layer<float> > >& layers = net->layers_;
  vector<vector<Blob<float>*> >& bottom_vecs = net->bottom_vecs_;
  vector<vector<Blob<float>*> >& top_vecs = net->top_vecs_;
  vector<shared_ptr<Blob<float> > >& imbs = net->blobs_;
  vector<string>& layer_types = net->layer_types_;
  vector<string>& layer_names = net->layer_names_;
  /* When we test on the testing network, we will use the layer information
   * that is gathered using training network, so we are assuming
   * the testing network has the same topology as the training network. */
  tbb::tick_count tick_start;

  bool print_ = this->ps_config_.debug ? ps_config_.worker_id == 0 : false;

  if (print_) {
    if (test) {
      LOG(INFO) << "TEST";
    } else {
      LOG(INFO) << "TRAIN";
    }
  }

  /* Forward */
  if (print_) {
    LOG(INFO) << "Forward";
  }
  float loss = 0;
  for (int batch_id = 0; batch_id < ps_config_.batches_per_clock; batch_id++) {
    for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
      if (print_) {
        LOG(INFO) << "Layer " << layer_id << ": " << layer_names[layer_id]
                  << "(" << layer_types[layer_id] << ")";
      }
      CHECK_LT(layer_id, layers.size());
      shared_ptr<Layer<float> >& layer = layers[layer_id];
      CHECK(layer);
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];

      tick_start = tbb::tick_count::now();
      /* Read model parameters */
      if (layer_info.param_infos.size()) {
        if (print_) {
          LOG(INFO) << "Read params: handle #" << layer_handles.read_handle;
        }
        RowData *read_buffer = NULL;
        if (!layer_info.local_param) {
          ps_->Read(layer_handles.read_handle, &read_buffer);
        } else {
          ps_->LocalAccess(layer_handles.read_handle, &read_buffer);
        }
        float *params_vals = reinterpret_cast<float *>(read_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &params_vals[param_val_offset];
          CHECK_LT(param_id, layer->blobs().size());
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          CHECK(!param->check_gpu_data())
              << "layer " << layer_names[layer_id] << " has gpu param";
          CHECK_EQ(param->check_data_head(), SyncedMemory::UNINITIALIZED);
          param->set_gpu_data(param_vals, true);
          if (do_snapshot) {
            /* Write the model parameter data to snapshot protobuf */
            tbb::tick_count snapshot_start;
            CHECK(!test);
            NetParameter& net_param_pb = this->snapshot_net_param_protobuf_;
            CHECK_LT(layer_id, net_param_pb.layer_size());
            LayerParameter *layer_pb = net_param_pb.mutable_layer(layer_id);
            CHECK_EQ(layer_names[layer_id], layer_pb->name());
            CHECK_LT(param_id, layer_pb->blobs_size());
            BlobProto *blob_pb = layer_pb->mutable_blobs(param_id);
            bool write_diff = false;
            bool write_data = true;
            param->ToProto(blob_pb, write_diff, write_data);
            layer_info.snapshot_model_time +=
                (tbb::tick_count::now() - snapshot_start).seconds();
          }
        }
      }
#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      if (print_) {
        LOG(INFO) << "Read intermediate data blobs";

        LOG(INFO) << "Reading forward data for layer " << layer_names[layer_id] 
                << ", num: " << layer_info.imbs_to_access_fw.size();
      }
      
      for (int i = 0; i < layer_info.imbs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_fw[i];
        CHECK_LT(i, layer_handles.imbs_to_access_fw.size());
        int handle = layer_handles.imbs_to_access_fw[i];
        if (print_) {
          LOG(INFO) << "Read data " << imb_info.global_imb_id;
        }
        CHECK_LT(imb_info.global_imb_id, imbs.size());
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowData *read_buffer = NULL;
        ps_->LocalAccess(handle, &read_buffer);
        CHECK(!imb->check_gpu_data())
            << "layer " << layer_names[layer_id] << " has gpu data "
            << imb_info.global_imb_id;
        imb->set_gpu_data(reinterpret_cast<float *>(read_buffer), true);
        if (print_) {
          LOG(INFO) << "Check blob #" << imb_info.global_imb_id
                    << " : " << imb->check_gpu_data();
        }
      }
      /* Access intermediate diff blobs */
      if (print_) {
        LOG(INFO) << "Read intermediate diff blobs";
      }
      for (int i = 0; i < layer_info.imb_diffs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_fw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_fw.size());
        int handle = layer_handles.imb_diffs_to_access_fw[i];
        if (print_) {
          LOG(INFO) << "Read data " << imb_info.global_imb_id;
        }
        CHECK_LT(imb_info.global_imb_id, imbs.size());
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowData *read_buffer = NULL;
        ps_->LocalAccess(handle, &read_buffer);
        CHECK(!imb->check_gpu_diff())
            << "layer " << layer_names[layer_id] << " has gpu diff";
        imb->set_gpu_diff(reinterpret_cast<float *>(read_buffer), true);
        if (print_) {
          LOG(INFO) << "Check blob diff #" << imb_info.global_imb_id
                    << " : " << imb->check_gpu_diff();
        }
      }
#endif
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.fw_read_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      } else {
        layer_info.test_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      }

      if (print_) {
        vector<int>& bottom_imb_ids = this->net_->bottom_id_vecs_[layer_id];
        vector<int>& top_imb_ids = this->net_->top_id_vecs_[layer_id];
        for (int i = 0; i < bottom_imb_ids.size(); i++) {
          shared_ptr<Blob<float> >& imb = imbs[bottom_imb_ids[i]];
          LOG(INFO) << "Check blob #" << bottom_imb_ids[i]
                    << " : " << imb->check_gpu_data();
          float blob_dot;
          caffe_gpu_dot<float>(imb->count(), imb->gpu_data(), imb->gpu_data(), &blob_dot);
          LOG(INFO) << "Blob #" << bottom_imb_ids[i]
                    << ", dot = " << blob_dot;
        }
        for (int i = 0; i < top_imb_ids.size(); i++) {
          LOG(INFO) << "Check blob #" << top_imb_ids[i]
                    << " : " << imbs[top_imb_ids[i]]->check_gpu_data();
        }

        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          float param_dot;
          caffe_gpu_dot<float>(param->count(), param->gpu_data(), param->gpu_data(), &param_dot);
          LOG(INFO) << "Param #" << param_id
                    << ", dot = " << param_dot;
        }
      }

      /* Forward calculation */
      if (print_) {
        LOG(INFO) << "Forward calculation";
      }
      tick_start = tbb::tick_count::now();
#if (PARAM_DBG)
      if (0 == ps_config_.worker_id) {
        LOG(INFO) << "Forward layer " << layer_names[layer_id];
      }
#endif
      float layer_loss =
          layer->Forward(bottom_vecs[layer_id], top_vecs[layer_id]);
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));

#if (PARAM_DBG)
      float print_buf[DBG_SIZE]; 
      for (int top_id = 0; top_id < top_vecs[layer_id].size(); ++top_id) {
        const Blob<float>& blob = *top_vecs[layer_id][top_id];
        int print_size = std::min(DBG_SIZE, blob.count());
        std::stringstream value_out;
        caffe_copy(print_size, blob.gpu_data(), print_buf);
        value_out << "[FW] Top " << top_id << " Values: ";
        for (int i = 0; i < print_size; i++) {
          value_out << print_buf[i] << ", ";
        }
        if (0 == ps_config_.worker_id) {
          LOG(INFO) << value_out.str();
        }
      }
      for (int param_id = 0; param_id < layer->blobs().size();
           ++param_id) {
        const Blob<float>& blob = *layer->blobs()[param_id];
        int net_param_id = layer_info.param_infos[param_id].global_param_id;
        int print_size = std::min(DBG_SIZE, blob.count());
        std::stringstream value_out;
        caffe_copy(print_size, blob.gpu_data(), print_buf);
        value_out << "[FW] Param " << net_param_id << " Values: ";
        for (int i = 0; i < print_size; i++) {
          value_out << print_buf[i] << ", ";
        }
        if (0 == ps_config_.worker_id) {
          LOG(INFO) << value_out.str();
        }
      }
#endif

      loss += layer_loss;
      if (!test) {
        layer_info.fw_compute_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      } else {
        layer_info.test_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      }

      tick_start = tbb::tick_count::now();
      /* We release the intermediate data first, because usually the
       * intermediate data released here was allocated from the last layer,
       * before the allocation of the parameter data of this layer. */
#if defined(LOCAL_DATA_IN_PS)
      /* Release intermediate data blobs */
      if (print_) {
        LOG(INFO) << "Release intermediate data blobs";
      }
      for (int i = 0; i < layer_info.imbs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_fw[i];
        CHECK_LT(i, layer_handles.imbs_to_release_fw.size());
        int handle = layer_handles.imbs_to_release_fw[i];
        if (print_) { //KEVIN DBG
          LOG(INFO) << "Release data " << imb_info.global_imb_id;
        }
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_data();
           /* Make sure everything is copied to GPU memory */
        imb->set_gpu_data(NULL, true);
        ps_->PostLocalAccess(handle);
        if (print_) {
          LOG(INFO) << "Release blob #" << imb_info.global_imb_id
                    << " : " << imb->check_gpu_data();
        }
      }
      /* Release intermediate diff blobs */
      if (print_) {
        LOG(INFO) << "Release intermediate diff blobs";
      }
      for (int i = 0; i < layer_info.imb_diffs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_fw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_release_fw.size());
        int handle = layer_handles.imb_diffs_to_release_fw[i];
        if (print_) {
          LOG(INFO) << "Release data " << imb_info.global_imb_id;
        }
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_diff();
           /* Make sure everything is copied to GPU memory */
        imb->set_gpu_diff(NULL, true);
        ps_->PostLocalAccess(handle);
        if (print_) {
          LOG(INFO) << "Release blob #" << imb_info.global_imb_id
                    << " : " << imb->check_gpu_data();
        }
      }

#endif
      /* Release read buffers */
      if (layer_info.param_infos.size()) {
        if (print_) {
          LOG(INFO) << "Release read buffers: handle #"
              << layer_handles.postread_handle;
        }
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          CHECK_NE(param->check_data_head(), SyncedMemory::HEAD_AT_CPU);
          param->set_gpu_data(NULL, true);
        }
        if (!layer_info.local_param) {
          ps_->PostRead(layer_handles.postread_handle);
        } else {
          ps_->PostLocalAccess(layer_handles.postread_handle);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.fw_write_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      } else {
        layer_info.test_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      }
    }

    /* Backward */
    if (print_) {
      LOG(INFO) << "Backward";
    }
    for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
      if (print_) {
        LOG(INFO) << "Layer " << layer_id << ": " << layer_names[layer_id];
      }
      CHECK_LT(layer_id, layers.size());
      shared_ptr<Layer<float> >& layer = layers[layer_id];
      CHECK(layer);
      LayerInfo& layer_info = layer_infos_[layer_id];
      if (!layer_info.layer_need_backward) {
        continue;
      }
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];

      tick_start = tbb::tick_count::now();
      if (layer_info.param_infos.size()) {
        /* Prepare write buffers */
        if (print_) {
          LOG(INFO) << "Prepare write buffers";
        }
        CHECK(!layer_info.local_param);
        RowData *write_buffer = NULL;
        ps_->PreUpdate(layer_handles.prewrite_handle, &write_buffer);
        float *write_params_vals = reinterpret_cast<float *>(write_buffer);
        size_t size = layer_info.num_vals * sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(
            write_params_vals, 0, size, Caffe::cuda_stream()));
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &write_params_vals[param_val_offset];
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_diff(param_vals, true);
          /* "true" means that we don't keep CPU data */
        }
        /* Read params */
        if (print_) {
          LOG(INFO) << "Read params";
        }
        RowData *read_buffer = NULL;
        ps_->Read(layer_handles.bw_read_handle, &read_buffer);
        float *read_params_vals = reinterpret_cast<float *>(read_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &read_params_vals[param_val_offset];
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_data(param_vals, true);
          if (print_) {
            LOG(INFO) << "Set param pointer:" << param_vals;
          }
        }
        /* Access local updates history */
        RowData *history_buffer = NULL;
        ps_->LocalAccess(layer_handles.history_access_handle, &history_buffer);
        float *history_vals = reinterpret_cast<float *>(history_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *history_param_vals = &history_vals[param_val_offset];
          int global_param_id =
              layer_info.param_infos[param_id].global_param_id;
          shared_ptr<Blob<float> >& updates_history = history_[global_param_id];
          updates_history->set_gpu_data(history_param_vals, true);
          if (do_snapshot) {
            /* Write the updates history data to solver state protobuf */
            tbb::tick_count snapshot_start;
            CHECK(!test);
            SolverState& solverstate_pb = this->snapshot_solver_state_protobuf_;
            CHECK_LT(global_param_id, solverstate_pb.history_size());
            BlobProto *history_pb =
                solverstate_pb.mutable_history(global_param_id);
            bool write_diff = false;
            bool write_data = true;
            updates_history->ToProto(history_pb, write_diff, write_data);
            layer_info.snapshot_solverstate_time +=
                (tbb::tick_count::now() - snapshot_start).seconds();
          }
        }
      }
#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      if (print_) {
        LOG(INFO) << "Access intermediate data blobs";
      }
      for (int i = 0; i < layer_info.imbs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_bw[i];
        CHECK_LT(i, layer_handles.imbs_to_access_bw.size());
        int handle = layer_handles.imbs_to_access_bw[i];
        if (print_) {
          LOG(INFO) << "Read data " << imb_info.global_imb_id;
        }
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowData *imb_buffer = NULL;
        ps_->LocalAccess(handle, &imb_buffer);
        CHECK(!imb->check_gpu_data())
            << "layer " << layer_names[layer_id] << " has gpu data";
        imb->set_gpu_data(reinterpret_cast<float *>(imb_buffer), true);
        if (print_) {
          LOG(INFO) << "Check blob #" << imb_info.global_imb_id << " " << imb
                    << " : " << imb->check_gpu_data();
        }
      }
      /* Access intermediate diff blobs */
      if (print_) {
        LOG(INFO) << "Access intermediate diff blobs";
      }
      for (int i = 0; i < layer_info.imb_diffs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_bw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_bw.size());
        int handle = layer_handles.imb_diffs_to_access_bw[i];
        if (print_) {
          LOG(INFO) << "Read diff " << imb_info.global_imb_id;
        }
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowData *imb_buffer = NULL;
        ps_->LocalAccess(handle, &imb_buffer);
        CHECK(!imb->check_gpu_diff())
            << "layer " << layer_names[layer_id] << " has gpu diff";
        imb->set_gpu_diff(reinterpret_cast<float *>(imb_buffer), true);
        if (print_) {
          LOG(INFO) << "Check blob diff #" << imb_info.global_imb_id << " " << imb
                    << " : " << imb->check_gpu_diff();
        }
      }

#endif
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.bw_read_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      } else {
        layer_info.test_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      }

      if (print_) {
        vector<int>& bottom_imb_ids = this->net_->bottom_id_vecs_[layer_id];
        vector<int>& top_imb_ids = this->net_->top_id_vecs_[layer_id];
        for (int i = 0; i < bottom_imb_ids.size(); i++) {
          shared_ptr<Blob<float> >& imb = imbs[bottom_imb_ids[i]];
          LOG(INFO) << "Check blob #" << bottom_imb_ids[i]
                    << " : " << imb->check_gpu_data();
#if (PARAM_DBG)          
          int print_size = std::min(DBG_SIZE, imb->count());
          float print_buf[DBG_SIZE]; 
          std::stringstream value_out;
          caffe_copy(print_size, imb->gpu_data(), print_buf);
          value_out << "Blob Values: ";
          for (int i = 0; i < print_size; i++) {
            value_out << print_buf[i] << ", ";
          }
          if (0 == ps_config_.worker_id) {
            LOG(INFO) << value_out.str();
          }
#else
          float blob_dot;
          caffe_gpu_dot<float>(imb->count(), imb->gpu_data(), imb->gpu_data(), &blob_dot);
          LOG(INFO) << "Blob #" << bottom_imb_ids[i]
                    << ", dot = " << blob_dot;
#endif
        }
        for (int i = 0; i < top_imb_ids.size(); i++) {
          if (layer_types[layer_id] == "Data" ||
              (layer_types[layer_id] == "LRN" && i > 0) ||
              (layer_types[layer_id] == "Pooling" && i > 0) ||
              (layer_types[layer_id] == "BatchNorm" && i > 0) ||
              (layer_types[layer_id] == "BatchReNorm" && i > 0) ||
              (layer_types[layer_id] == "Dropout" && i > 0)) {
            /* Do not use top diff blobs */
            continue;
          }
          shared_ptr<Blob<float> >& imb = imbs[top_imb_ids[i]];
          LOG(INFO) << "Check blob diff #" << top_imb_ids[i]
                    << " : " << imbs[top_imb_ids[i]]->check_gpu_diff()
                    << ", data:" << imbs[top_imb_ids[i]]->check_gpu_diff();

#if (PARAM_DBG)
          int print_size = std::min(DBG_SIZE, imb->count());
          float print_buf[DBG_SIZE]; 
          std::stringstream value_out;
          caffe_copy(print_size, imb->gpu_diff(), print_buf);
          value_out << "Blob diff Values: ";
          for (int i = 0; i < print_size; i++) {
            value_out << print_buf[i] << ", ";
          }
          if (0 == ps_config_.worker_id) {
            LOG(INFO) << value_out.str();
          }
#else
          float blob_dot;
          caffe_gpu_dot<float>(imb->count(), imb->gpu_diff(), imb->gpu_diff(), &blob_dot);
          LOG(INFO) << "Blob #" << top_imb_ids[i]
                    << ", count = " << imb->count()
                    << ", diff dot = " << blob_dot;
#endif
        }

        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];

#if (PARAM_DBG)
          int print_size = std::min(DBG_SIZE, param->count());
          float print_buf[DBG_SIZE]; 
          std::stringstream value_out;
          caffe_copy(print_size, param->gpu_data(), print_buf);
          value_out << "Param Values: ";
          for (int i = 0; i < print_size; i++) {
            value_out << print_buf[i] << ", ";
          }
          if (0 == ps_config_.worker_id) {
            LOG(INFO) << value_out.str();
          }
#else
          float param_dot;
          caffe_gpu_dot<float>(param->count(), param->gpu_data(), param->gpu_data(), &param_dot);
          LOG(INFO) << "Param #" << param_id
                    << ", dot = " << param_dot;
#endif
        }
      }

      if (!test) {
        /* Backward calculation */
        if (print_) {
          LOG(INFO) << "Backward calculation for layer " << layer_names[layer_id];

          LOG(INFO) << "Top Vectors Num: " << top_vecs[layer_id].size();
          for (int i = 0; i < top_vecs[layer_id].size(); i++) {
            LOG(INFO) << top_vecs[layer_id][i] 
                      << " " << top_vecs[layer_id][i]->check_gpu_data()
                      << " " << top_vecs[layer_id][i]->check_gpu_diff();
          }

          LOG(INFO) << "Bottom Vectors Num: " << bottom_vecs[layer_id].size();
          for (int i = 0; i < bottom_vecs[layer_id].size(); i++) {
            LOG(INFO) << bottom_vecs[layer_id][i]
                      << " " << bottom_vecs[layer_id][i]->check_gpu_data()
                      << " " << bottom_vecs[layer_id][i]->check_gpu_diff();
          }
        }

#if (PARAM_DBG)
        if (0 == ps_config_.worker_id) {
          LOG(INFO) << "Backward layer " << layer_names[layer_id];
        }
#endif

        tick_start = tbb::tick_count::now();
        layer->Backward(top_vecs[layer_id], layer_info.bottom_need_backward,
            bottom_vecs[layer_id]);
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));

#if (PARAM_DBG)
        float print_buf[DBG_SIZE]; 
        for (int bottom_id = 0; bottom_id < bottom_vecs[layer_id].size(); ++bottom_id) {
          if (!layer_info.bottom_need_backward[bottom_id]) { continue; }
          const Blob<float>& blob = *bottom_vecs[layer_id][bottom_id];
          int print_size = std::min(DBG_SIZE, blob.count());
          std::stringstream value_out;
          caffe_copy(print_size, blob.gpu_diff(), print_buf);
          value_out << "[BW] Bottom " << bottom_id << " Diff Values: ";
          for (int i = 0; i < print_size; i++) {
            value_out << print_buf[i] << ", ";
          }
          if (0 == ps_config_.worker_id) {
            LOG(INFO) << value_out.str();
          }
        }
        for (int param_id = 0; param_id < layer->blobs().size();
             ++param_id) {
          const Blob<float>& blob = *layer->blobs()[param_id];
          int net_param_id = layer_info.param_infos[param_id].global_param_id;
          int print_size = std::min(DBG_SIZE, blob.count());
          std::stringstream value_out;
          caffe_copy(print_size, blob.gpu_diff(), print_buf);
          value_out << "[BW] Param " << net_param_id << " Diff Values: ";
          for (int i = 0; i < print_size; i++) {
            value_out << print_buf[i] << ", ";
          }
          if (0 == ps_config_.worker_id) {
            LOG(INFO) << value_out.str();
          }
        }
#endif        
        layer_info.bw_compute_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      }

      tick_start = tbb::tick_count::now();


#if defined(LOCAL_DATA_IN_PS)
      /* Release intermediate data blobs */
      if (print_) {
        LOG(INFO) << "Release intermediate data blobs";
      }
      for (int i = 0; i < layer_info.imbs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_bw[i];
        CHECK_LT(i, layer_handles.imbs_to_release_bw.size());
        int handle = layer_handles.imbs_to_release_bw[i];
        if (print_) {
          LOG(INFO) << "Release data " << imb_info.global_imb_id;
        }
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_data();
          /* Make sure everything is copied to GPU memory */
        imb->set_gpu_data(NULL, true);
        ps_->PostLocalAccess(handle);
        if (print_) {
          LOG(INFO) << "Check blob #" << imb_info.global_imb_id
                    << " : " << imb->check_gpu_data();
        }
      }
      /* Release intermediate diff blobs */
      if (print_) {
        LOG(INFO) << "Release intermediate diff blobs";
      }
      for (int i = 0; i < layer_info.imb_diffs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_bw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_release_bw.size());
        int handle = layer_handles.imb_diffs_to_release_bw[i];
        if (print_) {
          LOG(INFO) << "Release diff " << imb_info.global_imb_id;
        }
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_diff();
          /* Make sure everything is copied to GPU memory */
        imb->set_gpu_diff(NULL, true);
        ps_->PostLocalAccess(handle);
        if (print_) {
          LOG(INFO) << "Check blob #" << imb_info.global_imb_id
                    << " : " << imb->check_gpu_data();
        }
      }
#endif
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.bw_write_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      } else {
        layer_info.test_time +=
            (tbb::tick_count::now() - tick_start).seconds();
      }

      if (layer_info.param_infos.size()) {
        // LOG(INFO) << "Finish writing";
        tick_start = tbb::tick_count::now();
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int global_param_id =
              layer_info.param_infos[param_id].global_param_id;
          if (!test) {
            /* Adjust gradient */
            float learning_rate = GetLearningRate();
            // Normalize(global_param_id);
            Regularize(global_param_id);
            // LOG(INFO) << "ComputeUpdateValue";
            ComputeUpdateValue(global_param_id, learning_rate);
            CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
          }
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->gpu_diff();
          if (print_) {
            float param_dot;
            caffe_gpu_dot<float>(param->count(), param->gpu_diff(), param->gpu_diff(), &param_dot);
            LOG(INFO) << "Param #" << global_param_id << ", diff dot = " << param_dot;
          }
            /* Make sure everything is copied to GPU memory */
          param->set_gpu_diff(NULL, true);
        }
        if (!test) {
          layer_info.bw_compute_time +=
              (tbb::tick_count::now() - tick_start).seconds();
        } else {
          layer_info.test_time +=
              (tbb::tick_count::now() - tick_start).seconds();
        }

        tick_start = tbb::tick_count::now();
        /* Apply updates to PS */
        ps_->Update(layer_handles.write_handle);
        /* Release read buffers */
        if (print_) {
          LOG(INFO) << "Release read buffers";
        }
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_data(NULL, true);
        }
        ps_->PostRead(layer_handles.bw_postread_handle);
        /* Release local updates history */
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int global_param_id =
              layer_info.param_infos[param_id].global_param_id;
          history_[global_param_id]->gpu_data();
            /* Make sure everything is copied to GPU memory */
          history_[global_param_id]->set_gpu_data(NULL, true);
        }
        ps_->PostLocalAccess(layer_handles.history_postaccess_handle);
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
        if (!test) {
          layer_info.bw_write_time +=
              (tbb::tick_count::now() - tick_start).seconds();
        } else {
          layer_info.test_time +=
              (tbb::tick_count::now() - tick_start).seconds();
        }
      }
    }
  }
  loss /= ps_config_.batches_per_clock;
  ps_->Clock();
  return loss;
}

template <typename Dtype>
void Solver<Dtype>::InitSnapshot() {
  InitNetParameterSnapshot();
  InitSolverStateSnapshot();
}

template <typename Dtype>
void Solver<Dtype>::InitNetParameterSnapshot() {
  /* Init snapshot protobuf for model parameters.
   * We will keep using this protobuf structure in future snapshots. */
  /* I just call the Net::ToProto() function,
   * and I don't need the Net::ToProto() function to
   * write blob data in this template protobuf. */
  bool write_diff = false;
  bool write_data = false;
  this->net_->ToProto(&snapshot_net_param_protobuf_, write_diff, write_data);
}

template <typename Dtype>
void SGDSolver<Dtype>::InitSolverStateSnapshot() {
  /* Init snapshot protobuf for solver states.
   * We will keep using this protobuf structure in future snapshots. */
  this->snapshot_solver_state_protobuf_.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    this->snapshot_solver_state_protobuf_.add_history();
  }
}

template <>
void Solver<double>::InitPs() {
  CHECK(0);
}

template <>
void SGDSolver<double>::InitPsValues() {
  CHECK(0);
}

template <>
double SGDSolver<double>::ForwardBackwardUsingPs(
    const vector<Blob<double>* > & bottom,
    const shared_ptr<Net<double> >& net, bool test, bool do_snapshot) {
  CHECK(0);
  return 0;
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  InitPsValues();
  InitSnapshot();

  double write_snapshot_time = 0.0;
  tbb::tick_count tick_start = tbb::tick_count::now();

  while (iter_ < (stop_iter + 1)) {
    /* We iterate iter_ to (stop_iter + 1),
     * because we need one more iteration to do testing and snapshotting. */
    const bool do_test =
        param_.test_interval() && iter_ % param_.test_interval() == 0
            && (iter_ > 0 || param_.test_initialization());
    const bool display = param_.display() && iter_ % param_.display() == 0;
    const bool do_snapshot = iter_ != start_iter
        && param_.snapshot() && iter_ % param_.snapshot() == 0;
    const bool print_ps_info = iter_ != start_iter
        && (iter_ % 1000 == 0 || iter_ == stop_iter);

    if (print_ps_info) {
      double total_time = (tbb::tick_count::now() - tick_start).seconds();
      double read_time = 0;
      double write_time = 0;
      double compute_time = 0;
      double test_time = 0;
      double snapshot_time = 0;
      for (int i = 0; i < layer_infos_.size(); i++) {
        read_time += layer_infos_[i].fw_read_time;
        read_time += layer_infos_[i].bw_read_time;
        write_time += layer_infos_[i].fw_write_time;
        write_time += layer_infos_[i].bw_write_time;
        compute_time += layer_infos_[i].fw_compute_time;
        compute_time += layer_infos_[i].bw_compute_time;
        test_time += layer_infos_[i].test_time;
        snapshot_time += layer_infos_[i].snapshot_model_time;
        snapshot_time += layer_infos_[i].snapshot_solverstate_time;
      }
      snapshot_time += write_snapshot_time;
      LOG(INFO) << "Read PS time: " << read_time;
      LOG(INFO) << "Write PS time: " << write_time;
      LOG(INFO) << "Compute time: " << compute_time;
      LOG(INFO) << "Test time: " << test_time;
      LOG(INFO) << "Snapshot time: " << snapshot_time;
      LOG(INFO) << "Total time: " << total_time;
			if (iter_ % 2000 == 0) {
				std::stringstream ostream;
				for (int i = 0; i < layer_infos_.size(); i++) {
					ostream << "Layer Read time: " << i 
									<< "," << layer_infos_[i].fw_read_time
									<< "," << layer_infos_[i].bw_read_time << endl;
				}
				LOG(INFO) << ostream.str();
			}
      // LOG(INFO) << "Per layer forwardbackward times:";
      // for (int i = 0; i < layer_infos_.size(); i++) {
        // cerr << i << "," << layer_infos_[i].fw_read_time
             // << "," << layer_infos_[i].fw_compute_time
             // << "," << layer_infos_[i].fw_write_time
             // << endl;
      // }
      // for (int i = layer_infos_.size() - 1; i >= 0; i--) {
        // cerr << i << "," << layer_infos_[i].bw_read_time
             // << "," << layer_infos_[i].bw_compute_time
             // << "," << layer_infos_[i].bw_write_time
             // << endl;
      // }
    }

    if (do_test) {
      TestAll();
    }
    // accumulate the loss and gradient
    Dtype loss = 0;
    CHECK_EQ(param_.iter_size(), 1);
    bool test = false;
    loss = ForwardBackwardUsingPs(bottom_vec, this->net_, test, do_snapshot);
    CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
    // average the loss across iterations for smoothed reporting
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss
          << " worker" << ps_config_.worker_id;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG(INFO) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }

    /* Write snapshot files */
    if (do_snapshot) {
      tbb::tick_count snapshot_start = tbb::tick_count::now();
      Snapshot();
      write_snapshot_time +=
          (tbb::tick_count::now() - snapshot_start).seconds();
    }

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;
  }
  string json_stats = ps_->GetStats();
  //cerr << json_stats << endl;
  LOG(INFO) << json_stats;
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  // if (param_.display() && iter_ % param_.display() == 0) {
    // Dtype loss;
    // net_->ForwardPrefilled(&loss);
    // LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  // }
  // if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    // TestAll();
  // }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestAll() {
  LOG(INFO) << "test_nets_.size() = " << test_nets_.size();
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    bool test = true;
    bool do_snapshot = false;
    Dtype iter_loss =
        ForwardBackwardUsingPs(bottom_vec, test_net, test, do_snapshot);
    const vector<Blob<Dtype>*>& result = test_net->net_output_blobs_;
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  CHECK(0);
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string snapshot_name = SnapshotFilename("");
  string model_filename =
      (boost::format("%s.caffemodel.%i") % snapshot_name % ps_config_.worker_id).str();
  NetParameter& net_param = snapshot_net_param_protobuf_;
  LOG(INFO) << "Snapshotting model to " << model_filename;
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  CHECK(0);
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    /* Allocate CPU memory and zerofy the data */
    history_[history_.size()-1]->mutable_cpu_data();
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_[update_.size()-1]->mutable_cpu_data();
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_[temp_.size()-1]->mutable_cpu_data();
  }
  /* I assume the values will be zerofied on allocation (in SyncedMemory)*/
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    if (this->net_->param_owners()[i] < 0) {
      sumsq_diff += net_params[i]->sumsq_diff();
    }
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      if (this->net_->param_owners()[i] < 0) {
        net_params[i]->scale_diff(scale_factor);
      }
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  const vector<float>& net_params_lr = this->net_->params_lr();

  if (FIX_LEARNING_RATE == net_params_lr[param_id]) {
    local_decay = 0;
  }

  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        CHECK(0);
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();

#if (PARAM_DBG)
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
#endif

  // XXX: I made the local learning rate negative, so that the updates will be
  // added to the parameter data instead of subtracted
  // Dtype local_rate = rate * net_params_lr[param_id];
  Dtype local_rate = -rate * net_params_lr[param_id];
  if (FIX_LEARNING_RATE == net_params_lr[param_id]) {
    local_rate = Dtype(1);
    momentum = Dtype(0);
  }

#if (PARAM_DBG)
  if (0 == this->ps_config_.worker_id) {
    LOG(INFO) << "Update Param " << param_id
              << ", Rate: " << local_rate
              << ", Momentum: " << momentum;
  }
  int print_size = std::min(DBG_SIZE, net_params[param_id]->count());
  Dtype print_buf[DBG_SIZE]; 
  std::stringstream value_out, diff_out, history_value_out;  
  caffe_copy(print_size, net_params[param_id]->gpu_data(), print_buf);
  value_out << "Param Values: ";
  for (int i = 0; i < print_size; i++) {
    value_out << print_buf[i] << ", ";
  }
  if (0 == this->ps_config_.worker_id) {
    LOG(INFO) << value_out.str();
  }
  caffe_copy(print_size, net_params[param_id]->gpu_diff(), print_buf);
  diff_out << "This Iter Diff Values: ";
  for (int i = 0; i < print_size; i++) {
    diff_out << print_buf[i] << ", ";
  }
  if (0 == this->ps_config_.worker_id) {
    LOG(INFO) << diff_out.str();
  }  
  caffe_copy(print_size, history_[param_id]->gpu_data(), print_buf);
  history_value_out << "History Values: ";
  for (int i = 0; i < print_size; i++) {
    history_value_out << print_buf[i] << ", ";
  }
  if (0 == this->ps_config_.worker_id) {
    LOG(INFO) << history_value_out.str();
  }

#endif

#if (HISTORY_IN_SERVER)

  this->ps_->SetParamLearningRate(param_id, local_rate);
  this->ps_->SetParamMomentum(param_id, momentum);

#else
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              history_[param_id]->mutable_gpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
#endif // HISTORY_IN_SERVER

#if (PARAM_DBG)
  std::stringstream diff_value_out;
  caffe_copy(print_size, net_params[param_id]->gpu_diff(), print_buf);
  diff_value_out << "Diff Values: ";
  for (int i = 0; i < print_size; i++) {
    diff_value_out << print_buf[i] << ", ";
  }
  if (0 == this->ps_config_.worker_id) {
    LOG(INFO) << diff_value_out.str();
  }
#endif

}


template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  /* Actually this function can be moved to the parent Solver class */
  SolverState& state = this->snapshot_solver_state_protobuf_;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  string snapshot_name = this->SnapshotFilename("");
  string state_filename = (boost::format("%s.solverstate.%i")
      % snapshot_name % this->ps_config_.worker_id).str();
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << state_filename;
  WriteProtoToBinaryFile(state, state_filename);
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  CHECK(0);
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& snapshot_name) {
  SolverState state;
  string state_filename = (boost::format("%s.solverstate.%i")
      % snapshot_name % this->ps_config_.worker_id).str();
  ReadProtoFromBinaryFile(state_filename, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    string model_filename = (boost::format("%s.caffemodel")
        % snapshot_name).str();
    ReadNetParamsFromBinaryFileOrDie(model_filename.c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  if (this->ps_config_.keep_momentum) {
    CHECK_EQ(state.history_size(), history_.size())
        << "Incorrect length of history blobs.";
    LOG(INFO) << "SGDSolver: restoring history";
    for (int i = 0; i < history_.size(); ++i) {
      history_[i]->FromProto(state.history(i));
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  CHECK(0);
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
