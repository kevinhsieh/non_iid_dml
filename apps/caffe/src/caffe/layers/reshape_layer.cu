#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom[0]->count(), top[0]->count());
  /* XXX: copy instead of share, because the bottom data could be released */
  int count_ = bottom[0]->count();
  caffe_copy(count_, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom[0]->count(), top[0]->count());
  if (!propagate_down[0]) {
    return;
  }
  int count_ = bottom[0]->count();
  caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}


INSTANTIATE_LAYER_GPU_FUNCS(ReshapeLayer);

}  // namespace caffe
