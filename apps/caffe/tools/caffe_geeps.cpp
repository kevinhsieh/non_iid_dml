#include <boost/program_options.hpp>
#include <boost/format.hpp>

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "geeps.hpp"
#include "caffe/caffe.hpp"

namespace po = boost::program_options;

using std::vector;
using std::string;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

DEFINE_int32(worker_id, 0,
    "");
DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(machinefile, "",
    "Machine file path.");
DEFINE_string(ps_config, "",
    "Configuration file path.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(outdir, "",
    "Optional; the output dir.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

void parse_hostfile(const string& hostfile, vector<string>& hostlist) {
  std::ifstream is(hostfile.c_str());
  CHECK(is);
  std::string line;
  hostlist.clear();
  while (!!getline(is, line)) {
    hostlist.push_back(line);
  }
  is.close();
}

void parse_config_file(caffe::PsConfig& ps_config) {
  po::options_description desc("Allowed options");
  string gaia_threshold_sched;
  desc.add_options()
    ("slack",
     po::value<int>(&ps_config.slack),
     "")
    ("num_channels",
     po::value<uint32_t>(&(ps_config.geeps_config.num_comm_channels)),
     "")
    ("mm_warning_level",
     po::value<int>(&(ps_config.geeps_config.mm_warning_level))
     ->default_value(0),
     "")
    ("gpu_memory_capacity",
     po::value<size_t>(&(ps_config.geeps_config.gpu_memory_capacity))
     ->default_value(std::numeric_limits<size_t>::max()),
     "")
    ("read_my_writes",
     po::value<int>(&(ps_config.geeps_config.read_my_writes))
     ->default_value(0),
     "")
    ("pinned_cpu_memory",
     po::value<int>(&(ps_config.geeps_config.pinned_cpu_memory))
     ->default_value(1),
     "")
    ("batches_per_clock",
     po::value<int>(&(ps_config.batches_per_clock))
     ->default_value(1),
     "")
    ("multi_table",
     po::value<int>(&(ps_config.multi_table))
     ->default_value(1),
     "")
    ("layers_per_table",
     po::value<int>(&(ps_config.layers_per_table))
     ->default_value(1),
     "")
    ("restore_snapshot",
     po::value<string>(&(ps_config.snapshot_name))
     ->default_value(""),
     "")
    ("keep_momentum",
     po::value<int>(&(ps_config.keep_momentum))
     ->default_value(1),
     "")
    ("debug",
     po::value<int>(&(ps_config.debug))
     ->default_value(0),
     "")
    ("log_interval",
     po::value<int>(&(ps_config.geeps_config.log_interval))
     ->default_value(0),
     "")
    ("enable_gaia",
     po::value<int>(&(ps_config.geeps_config.enable_gaia))
     ->default_value(0),
     "")
    ("num_dc",
     po::value<int>(&(ps_config.geeps_config.num_dc))
     ->default_value(1),
     "")
	  ("mirror_update_threshold",
     po::value<float>(&(ps_config.geeps_config.mirror_update_threshold))
     ->default_value(0),
     "")
    ("merge_local_update",
     po::value<int>(&(ps_config.geeps_config.merge_local_update))
     ->default_value(0),
     "")
    ("aggr_mirror_update_table_group",
     po::value<int>(&(ps_config.geeps_config.aggr_mirror_update_table_group))
     ->default_value(0),
     "")
    ("aggr_mirror_update_threshold",
     po::value<float>(&(ps_config.geeps_config.aggr_mirror_update_threshold))
     ->default_value(0),
     "")
    ("enable_mirror_reorder",
     po::value<int>(&(ps_config.geeps_config.enable_mirror_reorder))
     ->default_value(0),
     "")
    ("wan_bandwidth_limit",
     po::value<float>(&(ps_config.geeps_config.wan_bandwidth_limit))
     ->default_value(0),
     "")
    ("slack_table_limit",
     po::value<int>(&(ps_config.geeps_config.slack_table_limit))
     ->default_value(-1),
     "")
    ("mirror_update_lower_bound",
     po::value<float>(&(ps_config.geeps_config.mirror_update_lower_bound))
     ->default_value(0),
     "")
    ("iters_reach_lower_bound",
     po::value<int>(&(ps_config.geeps_config.iters_reach_lower_bound))
     ->default_value(-1),
     "")
    ("enable_overlay_network",
     po::value<int>(&(ps_config.geeps_config.enable_overlay_network))
     ->default_value(0),
     "")
		("enable_olnw_multiple_routers",
     po::value<int>(&(ps_config.geeps_config.enable_olnw_multiple_routers))
     ->default_value(0),
     "")
    ("enable_threshold_by_bw",
     po::value<int>(&(ps_config.geeps_config.enable_threshold_by_bw))
     ->default_value(0),
     "")
    ("flush_mirror_update_per_iter",
     po::value<int>(&(ps_config.geeps_config.flush_mirror_update_per_iter))
     ->default_value(0),
     "")
    ("local_model_only",
     po::value<int>(&(ps_config.geeps_config.local_model_only))
     ->default_value(0),
     "")
    ("mirror_update_value_threshold",
     po::value<float>(&(ps_config.geeps_config.mirror_update_value_threshold))
     ->default_value(0),
     "")
    ("lower_update_threshold",
     po::value<float>(&(ps_config.geeps_config.lower_update_threshold))
     ->default_value(0),
     "")
    ("lower_update_table_limit",
     po::value<int>(&(ps_config.geeps_config.lower_update_table_limit))
     ->default_value(0),
     "")
    ("model_traveling_freq",
     po::value<int>(&(ps_config.geeps_config.model_traveling_freq))
     ->default_value(0),
     "")
    ("enable_fedavg",
     po::value<int>(&(ps_config.geeps_config.enable_fedavg))
     ->default_value(0),
     "")
    ("fedavg_local_iter",
     po::value<int>(&(ps_config.geeps_config.fedavg_local_iter))
     ->default_value(1),
     "")
    ("enable_dgc",
     po::value<int>(&(ps_config.geeps_config.enable_dgc))
     ->default_value(0),
     "")
    ("dgc_epoch_size",
     po::value<int>(&(ps_config.geeps_config.dgc_epoch_size))
     ->default_value(1000),
     "")
    ("apply_change_to_local_model",
     po::value<int>(&(ps_config.geeps_config.apply_change_to_local_model))
     ->default_value(0),
     "")
    ("disable_dgc_momemtum_mask",
     po::value<int>(&(ps_config.geeps_config.disable_dgc_momemtum_mask))
     ->default_value(0),
     "")
    ("gaia_threshold_sched",
     po::value<string>(&(gaia_threshold_sched))
     ->default_value(""),
     "")
    ("gaia_threshold_iter",
     po::value<int>(&(ps_config.geeps_config.gaia_threshold_iter))
     ->default_value(1),
     "")
    ("gradient_clip_threshold",
     po::value<int>(&(ps_config.geeps_config.gradient_clip_threshold))
     ->default_value(0),
     "")
    ;

  std::ifstream config_in(FLAGS_ps_config.c_str());
  CHECK(config_in);
  po::variables_map vm;
  po::store(po::parse_config_file(config_in, desc), vm);
  po::notify(vm);

  ps_config.geeps_config.output_dir = FLAGS_outdir;
  /* Cannot enable Gaia, FedAvg, DGC at the same time */
  CHECK_EQ(ps_config.geeps_config.enable_fedavg && ps_config.geeps_config.enable_gaia,
           0);
  CHECK_EQ(ps_config.geeps_config.enable_fedavg && ps_config.geeps_config.enable_dgc,
           0);
  CHECK_EQ(ps_config.geeps_config.enable_dgc && ps_config.geeps_config.enable_gaia,
           0);
  /* Local iteration must be a positive number for FedAvg */
  if (ps_config.geeps_config.enable_fedavg)
  {
    CHECK_GT(ps_config.geeps_config.fedavg_local_iter, 0);
  }

  /* epoch size must be a positive number for DGC */
  if (ps_config.geeps_config.enable_dgc)
  {
    CHECK_GT(ps_config.geeps_config.dgc_epoch_size, 0);
  }
           
  ps_config.geeps_config.enable_decentral = ps_config.geeps_config.enable_fedavg ||
    ps_config.geeps_config.enable_gaia || ps_config.geeps_config.enable_dgc;

  /* Make Gaia threshold schedule into a vector */
  if (gaia_threshold_sched != "") 
  {
    size_t pos_start = 0, pos_end;
    string token;
    while ((pos_end = gaia_threshold_sched.find(",", pos_start)) != string::npos) {
      token = gaia_threshold_sched.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + 1;
      ps_config.geeps_config.gaia_threshold_sched.push_back(std::atof(token.c_str()));
    }
    ps_config.geeps_config.gaia_threshold_sched.push_back(
      std::atof(gaia_threshold_sched.substr(pos_start).c_str()));
  }

	if (ps_config.geeps_config.enable_overlay_network)
	{
		GeePsConfig& config = ps_config.geeps_config;

		/* Hard code data center group info for 11 DCs */
		CHECK_EQ(config.num_dc, 11);

		config.dcg_id_list.resize(config.num_dc);
		config.dcg_router_list.resize(config.num_dc, false);

		config.dcg_id_list[0] = 0; /* Virginia */
		config.dcg_router_list[0] = true;
		config.dcg_id_list[1] = 0; /* California */
		config.dcg_id_list[2] = 0; /* Oregon */
		config.dcg_id_list[10] = 0; /* Sao Paulo */

		config.dcg_id_list[5] = 1; /* Tokyo */
		config.dcg_router_list[5] = true;
		config.dcg_id_list[6] = 1; /* Seoul */
		config.dcg_id_list[7] = 1; /* Singapore */
		config.dcg_id_list[8] = 1; /* Sydney */

		config.dcg_id_list[4] = 2; /* Frankfurt */
		config.dcg_router_list[4] = true;
		config.dcg_id_list[3] = 2; /* Ireland */
		config.dcg_id_list[9] = 2; /* Mumbai */		

		config.num_dcg = 3;

		if (config.enable_olnw_multiple_routers)
		{
			config.dcg_router_list.clear();
			config.dcg_router_list.resize(config.num_dc, false);
			config.dcg_peer_router_list.resize(config.num_dc, 0);
			config.dcg_router_list[0] = true;
			config.dcg_peer_router_list[0] = 4; /* Virginia <-> Frankfurt */

			config.dcg_router_list[2] = true;
			config.dcg_peer_router_list[2] = 6; /* Oregon <-> Seoul */

			config.dcg_router_list[5] = true;
			config.dcg_peer_router_list[5] = 9; /* Tokyo <-> Mumbai */

			config.dcg_router_list[6] = true;
			config.dcg_peer_router_list[6] = 2; /* Oregon <-> Seoul */

			config.dcg_router_list[4] = true;
			config.dcg_peer_router_list[4] = 0; /* Virginia <-> Frankfurt */
			
			config.dcg_router_list[9] = true;
			config.dcg_peer_router_list[9] = 5; /* Tokyo <-> Mumbai */			
		}
	}
}

// Train / Finetune a model.
int train() {
  // XXX: change solver file name according to the worker_id
  int worker_id = FLAGS_worker_id;
  FLAGS_solver = (boost::format("%s.%i") % FLAGS_solver % worker_id).str();
  LOG(INFO) << "Use solver " << FLAGS_solver;

  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  /* XXX: prepare PS config */
  caffe::PsConfig ps_config;
  parse_config_file(ps_config);
  parse_hostfile(FLAGS_machinefile, ps_config.geeps_config.host_list);
  ps_config.num_workers = ps_config.geeps_config.host_list.size();
  CHECK_LT(worker_id, ps_config.num_workers);
  ps_config.worker_id = worker_id;
  ps_config.geeps_config.lr_policy = solver_param.lr_policy();
  ps_config.geeps_config.max_iter = solver_param.max_iter();
  ps_config.geeps_config.lr_gamma = solver_param.gamma();
  ps_config.geeps_config.lr_power = solver_param.power();
  ps_config.geeps_config.lr_stepsize = solver_param.stepsize();
  ps_config.geeps_config.momentum = solver_param.momentum();
  for (int s = 0; s < solver_param.stepvalue_size(); s++) {
    ps_config.geeps_config.lr_stepvalue.push_back(solver_param.stepvalue(s));
  }
  Caffe::set_worker_id(worker_id);

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(
        solver_param, &ps_config));

  if (FLAGS_snapshot.size()) {
    ps_config.snapshot_name = FLAGS_snapshot;
  }
  if (ps_config.snapshot_name.size()) {
    LOG(INFO) << "Resuming from " << ps_config.snapshot_name;
    solver->Solve(ps_config.snapshot_name);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // FLAGS_alsologtostderr = 0;

  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
