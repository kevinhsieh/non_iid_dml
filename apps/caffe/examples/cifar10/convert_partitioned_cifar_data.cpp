//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "boost/format.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
                     const string& db_type, const string& setting, int num_parts, int part_id) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  string train_db_name = (boost::format("%s/cifar10_train_%s.%d") % output_folder % db_type % part_id).str();
  train_db->Open(train_db_name, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  bool iid = (setting == "shuffled");
  bool skewed_80 = (setting == "skewed_80");
  int alt_part_id = (part_id % 2 == 0) ? (part_id + 1) : (part_id - 1);
  int rand_value;

  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);
  int count = 0;

  LOG(INFO) << "Writing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    string batchFileName = input_folder + "/data_batch_"
      + caffe::format_int(fileid+1) + ".bin";
    std::ifstream data_file(batchFileName.c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);

      if (iid) {
        if (count++ % num_parts != part_id) // partition randomly
          continue;
      } else {
        if (!skewed_80){
          if (label % num_parts != part_id)  //partition by label
            continue;
        } else {
          rand_value = itemid % 10;
          if (rand_value < 8) {
              if (label % num_parts != part_id)  //partition by label
                continue;
          } else {
              if (label % num_parts != alt_part_id)  //partition by label using alt part ID
                continue;
          }
        }
      }
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
    }
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  count = 0;
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  string test_db_name = (boost::format("%s/cifar10_test_%s.%d") % output_folder % db_type % part_id).str();
  test_db->Open(test_db_name, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    if (count++ % num_parts != part_id) {
      continue;
    }
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();
}

int main(int argc, char** argv) {
  if (argc != 7) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type setting num_part part_id\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);    
    int num_parts = atoi(argv[5]);
    int part_id = atoi(argv[6]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]), num_parts, part_id);
  }
  return 0;
}
