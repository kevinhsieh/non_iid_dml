#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MEAN_ONLY (0)
#define PRINT_DATA (0)

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

  /* XXX: use top[1] and top[2] for temp and x_norm */
  Blob<Dtype>& temp = *top[1];
  Blob<Dtype>& x_norm = *top[2];

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }


  if (use_global_stats_) {
    Dtype m_counter; 
    caffe_copy(1, this->blobs_[2]->gpu_data(), &m_counter);

    const Dtype scale_factor = m_counter == 0 ?
        0 : 1 / m_counter;

    if (0 == Caffe::worker_id()) {
      LOG(INFO) << "m_counter:" << m_counter;
    }

    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    // compute mean
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());
  }

  // subtract mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    //caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
    //    temp.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_mul(top[0]->count(), top[0]->gpu_data(), top[0]->gpu_data(),
        temp.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        variance_.mutable_gpu_data());  // E((X_EX)^2)
    // Backup the variance here so we can calculate moving average at
    // the backward phase
    caffe_copy(variance_.count(), variance_.gpu_data(), 
               variance_back_.mutable_gpu_data());

    // compute and save moving average
    /* XXX: I don't want to use the global stats */
    //this->blobs_[2]->mutable_gpu_data()[0] *= moving_average_fraction_;
    //this->blobs_[2]->mutable_gpu_data()[0] += 1;
    //caffe_gpu_scal<Dtype>(this->blobs_[2]->count(), moving_average_fraction_, 
    //                      this->blobs_[2]->mutable_gpu_data());
    //caffe_gpu_add_scalar<Dtype>(this->blobs_[2]->count(), 1.,
    //                            this->blobs_[2]->mutable_gpu_data());
    //caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
    //    moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    //int m = bottom[0]->count()/channels_;
    //Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    //caffe_gpu_axpby(variance_.count(), bias_correction_factor,
    //    variance_.gpu_data(), moving_average_fraction_,
    //    this->blobs_[1]->mutable_gpu_data());
  }

  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(),
      variance_.mutable_gpu_data());
  //caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
  //    variance_.mutable_gpu_data());

  // replicate variance to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp.mutable_gpu_data());

#if !MEAN_ONLY
  caffe_gpu_div(temp.count(), top_data, temp.gpu_data(), top_data);
#endif

#if PRINT_DATA
  if (!use_global_stats_) {
    Dtype out_buf[channels_];
    caffe_copy(channels_, mean_.gpu_data(), out_buf);
    for (int i = 0; i < channels_; i++) {
      LOG(INFO) << "mean:" << out_buf[i];
    }
    caffe_copy(channels_, variance_.gpu_data(), out_buf);
    for (int i = 0; i < channels_; i++) {
      LOG(INFO) << "variance:" << out_buf[i];
    }
  }
#endif  

  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm.count(), top_data,
      x_norm.mutable_gpu_data());
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  /* XXX: use top[1] and top[2] for temp and x_norm */
  Blob<Dtype>& temp = *top[1];
  Blob<Dtype>& x_norm = *top[2];

  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(x_norm.count(), top[0]->gpu_diff(), x_norm.mutable_gpu_diff());
    top_diff = x_norm.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  if (use_global_stats_) {
    caffe_gpu_div(temp.count(), top_diff, temp.gpu_data(), bottom_diff);
    return;
  }

  // compute and save moving average
  if (!use_global_stats_) {
    const Dtype* m_counter = this->blobs_[2]->gpu_data();    
    Dtype* m_counter_diff = this->blobs_[2]->mutable_gpu_diff();    

    const Dtype* m_average = this->blobs_[0]->gpu_data();
    Dtype* m_average_diff = this->blobs_[0]->mutable_gpu_diff();    

    const Dtype* m_variance = this->blobs_[1]->gpu_data();
    Dtype* m_variance_diff = this->blobs_[1]->mutable_gpu_diff();
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;

    // We only apply moving average fraction on worker 0
    //if (0 == Caffe::worker_id()) {
    {
      caffe_gpu_axpy(this->blobs_[2]->count(), moving_average_fraction_ - Dtype(1),
                     m_counter, m_counter_diff);
      caffe_gpu_axpy(mean_.count(), moving_average_fraction_ - Dtype(1), 
                     m_average, m_average_diff);
      caffe_gpu_axpy(variance_back_.count(), moving_average_fraction_ - Dtype(1), 
                     m_variance, m_variance_diff);
    }

    caffe_gpu_add_scalar(this->blobs_[2]->count(), Dtype(1), m_counter_diff);
    caffe_gpu_axpy(mean_.count(), Dtype(1), mean_.gpu_data(), m_average_diff);
    caffe_gpu_axpy(variance_back_.count(), bias_correction_factor, 
                   variance_back_.gpu_data(), m_variance_diff);

    //if (1 == Caffe::worker_id()) {
    //}

    /* XXX: I don't want to use the global stats */
    //this->blobs_[2]->mutable_gpu_data()[0] *= moving_average_fraction_;
    //this->blobs_[2]->mutable_gpu_data()[0] += 1;
    //caffe_gpu_scal<Dtype>(this->blobs_[2]->count(), moving_average_fraction_, 
    //                      this->blobs_[2]->mutable_gpu_data());
    //caffe_gpu_add_scalar<Dtype>(this->blobs_[2]->count(), 1.,
    //                            this->blobs_[2]->mutable_gpu_data());
    //caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
    //    moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    //int m = bottom[0]->count()/channels_;
    //Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    //caffe_gpu_axpby(variance_.count(), bias_correction_factor,
    //    variance_.gpu_data(), moving_average_fraction_,
    //    this->blobs_[1]->mutable_gpu_data());
  }


  const Dtype* top_data = x_norm.gpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(temp.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(temp.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());

#if MEAN_ONLY
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);
#else
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);
#endif

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(temp.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp still contains sqrt(var(X)+eps), computed during the forward
  // pass.
#if !MEAN_ONLY
  caffe_gpu_div(temp.count(), bottom_diff, temp.gpu_data(), bottom_diff);
#endif

#if 0
  if (!use_global_stats_) {
    Dtype m_counter, m_counter_diff; 
    caffe_copy(1, this->blobs_[2]->gpu_data(), &m_counter);
    caffe_copy(1, this->blobs_[2]->mutable_gpu_diff(), &m_counter_diff);
    LOG(INFO) << "m_counter:" << m_counter << ", diff:" << m_counter_diff;
    Dtype m_avg, m_avg_diff; 
    caffe_copy(1, this->blobs_[0]->gpu_data(), &m_avg);
    caffe_copy(1, this->blobs_[0]->mutable_gpu_diff(), &m_avg_diff);
    LOG(INFO) << "m_avg:" << m_avg << ", diff:" << m_avg_diff;
    //caffe_gpu_set(1, Dtype(0), this->blobs_[2]->mutable_gpu_diff());
  }
#endif

}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);


}  // namespace caffe
