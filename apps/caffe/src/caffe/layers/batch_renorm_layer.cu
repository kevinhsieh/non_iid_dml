#include <algorithm>
#include <vector>

#include "caffe/layers/batch_renorm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void R_D_CUT(const int n, Dtype* r, Dtype* d
		, Dtype cur_r_max, Dtype cur_r_min, Dtype cur_d_max, Dtype cur_d_min) {
		CUDA_KERNEL_LOOP(index, n) {
			r[index] = __min(cur_r_max, __max(r[index], cur_r_min));
			d[index] = __min(cur_d_max, __max(d[index], cur_d_min));
		}
	}

	template <typename Dtype>
	void BatchReNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int num = bottom[0]->shape(0);
		int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
		Dtype iter;
    caffe_copy(1, this->blobs_[3]->gpu_data(), &iter);
		int step = int(iter) / iter_size_;
		bool first_iter_in_step = (int(iter)%iter_size_ == 0);

    //LOG(INFO) << this->layer_param_.name() << " iter:" << iter << ", step:" << step;

    /* Use top[1], top[2], top[3], and top[4] for temp, x_norm, all_r, and all_d */
    Blob<Dtype>& temp = *top[1];
    Blob<Dtype>& x_norm = *top[2];
    Blob<Dtype>& all_r = *top[3];
    Blob<Dtype>& all_d = *top[4];
    

		if (bottom[0] != top[0]) {
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
		}

		if (use_global_stats_) {
			// use the stored mean/variance estimates.
      Dtype m_counter; 
      caffe_copy(1, this->blobs_[2]->gpu_data(), &m_counter);
      const Dtype scale_factor = m_counter == 0 ? 0 : 1 / m_counter;
			caffe_gpu_scale(variance_.count(), scale_factor,
				this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
			caffe_gpu_scale(variance_.count(), scale_factor,
				this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
      if (0 == Caffe::worker_id()) {
        LOG(INFO) << this->layer_param_.name() << " iter:" << iter << ", m_counter:" << m_counter;
      }
		}
		else {
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
			caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
				temp.mutable_gpu_data());  // (X-EX)^2
			caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), temp.gpu_data(),
				spatial_sum_multiplier_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data());
			caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
				variance_.mutable_gpu_data());  // E((X_EX)^2)

			if (step >= step_to_init_ && first_iter_in_step)
			{
        Dtype m_counter; 
        caffe_copy(1, this->blobs_[2]->gpu_data(), &m_counter);
				const Dtype scale_factor = 1. / m_counter;
				caffe_gpu_scale(variance_.count(), scale_factor, this->blobs_[0]->gpu_data(), 
                        mean_glb_.mutable_gpu_data());
				caffe_gpu_scale(variance_.count(), scale_factor, this->blobs_[1]->gpu_data(), 
                        variance_glb_.mutable_gpu_data());
				caffe_gpu_add_scalar(variance_.count(), eps_, variance_glb_.mutable_gpu_data());
				caffe_gpu_powx(variance_.count(), this->variance_glb_.gpu_data(), Dtype(0.5), 
                       this->variance_glb_.mutable_gpu_data());
			}

      // Backup the variance here so we can calculate moving average at
      // the backward phase
      caffe_copy(variance_.count(), variance_.gpu_data(), 
                 variance_back_.mutable_gpu_data());
			// compute and save moving average
			//Dtype moving_average_fraction = first_iter_in_step ? moving_average_fraction_ : 1.0;
			//this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction;
			//this->blobs_[2]->mutable_cpu_data()[0] += 1;
			//caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
			//	moving_average_fraction, this->blobs_[0]->mutable_gpu_data());
			//int m = bottom[0]->count() / channels_;
			//Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
			//caffe_gpu_axpby(variance_.count(), bias_correction_factor,
			//	variance_.gpu_data(), moving_average_fraction,
			//	this->blobs_[1]->mutable_gpu_data());
		}

		// normalize variance
		caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
		caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
			variance_.mutable_gpu_data());

		// replicate variance to input size
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), 0., temp.mutable_gpu_data());
		caffe_gpu_div(temp.count(), top_data, temp.gpu_data(), top_data);
		// TODO(cdoersch): The caching is only needed because later in-place layers
		//                 might clobber the data.  Can we skip this if they won't?
		caffe_copy(x_norm.count(), top_data,
			x_norm.mutable_gpu_data());

		if (!use_global_stats_ && step >= step_to_init_)
		{
			Dtype cur_r_max = __max(1, __min(1 + (step - step_to_init_ + 1)*(r_max_ - 1) / (step_to_r_max_ - step_to_init_), r_max_));
			Dtype cur_r_min = 1. / cur_r_max;
			Dtype cur_d_max = __max(0, __min((step - step_to_init_ + 1)*d_max_ / (step_to_d_max_ - step_to_init_), d_max_));
			Dtype cur_d_min = -cur_d_max;

			caffe_gpu_div(variance_.count(), variance_.gpu_data(), variance_glb_.gpu_data(), 
                    r_.mutable_gpu_data());

			caffe_copy(variance_.count(), mean_.gpu_data(), d_.mutable_gpu_data());
			caffe_gpu_axpby(variance_.count(), Dtype(-1), mean_glb_.gpu_data(), 
                      Dtype(1), d_.mutable_gpu_data());
			caffe_gpu_div(variance_.count(), d_.gpu_data(), variance_glb_.gpu_data(), 
                    d_.mutable_gpu_data());

			R_D_CUT<Dtype> << <CAFFE_GET_BLOCKS(variance_.count()), CAFFE_CUDA_NUM_THREADS >> >(
				variance_.count(), r_.mutable_gpu_data(), d_.mutable_gpu_data(), cur_r_max, cur_r_min
				, cur_d_max, cur_d_min);
			CUDA_POST_KERNEL_CHECK;

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
				batch_sum_multiplier_.gpu_data(), r_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data());
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
				spatial_dim, 1, 1., num_by_chans_.gpu_data(),
				spatial_sum_multiplier_.gpu_data(), 0., all_r.mutable_gpu_data());
			caffe_gpu_mul(temp.count(), top_data, all_r.gpu_data(), top_data);

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
				batch_sum_multiplier_.gpu_data(), d_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data());
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
				spatial_dim, 1, 1., num_by_chans_.gpu_data(),
				spatial_sum_multiplier_.gpu_data(), 0., all_d.mutable_gpu_data());
			caffe_gpu_add(temp.count(), top_data, all_d.gpu_data(), top_data);
		}
	}

	template <typename Dtype>
	void BatchReNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff;

    /* Use top[1], top[2], top[3], and top[4] for temp, x_norm, all_r, and all_d */
    Blob<Dtype>& temp = *top[1];
    Blob<Dtype>& x_norm = *top[2];
    Blob<Dtype>& all_r = *top[3];
    Blob<Dtype>& all_d = *top[4];

		Dtype iter;
    caffe_copy(1, this->blobs_[3]->gpu_data(), &iter);
		int step = int(iter) / iter_size_;

		if (bottom[0] != top[0]) {
			top_diff = top[0]->gpu_diff();
		}
		else {
			caffe_copy(x_norm.count(), top[0]->gpu_diff(), x_norm.mutable_gpu_diff());
			top_diff = x_norm.gpu_diff();
		}
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		if (use_global_stats_)
		{
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
    }

		const Dtype* top_data = x_norm.gpu_data();
		int num = bottom[0]->shape()[0];
		int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
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
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
			spatial_dim, 1, 1., num_by_chans_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

		// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
		caffe_gpu_axpby(temp.count(), Dtype(1), top_diff,
			Dtype(-1. / (num * spatial_dim)), bottom_diff);

		// note: temp still contains sqrt(var(X)+eps), computed during the forward
		// pass.
		caffe_gpu_div(temp.count(), bottom_diff, temp.gpu_data(), bottom_diff);
		

		if (!use_global_stats_ && step >= step_to_init_)
		{
			caffe_gpu_mul(temp.count(), bottom_diff, all_r.gpu_data(), bottom_diff);
		}

		if (this->phase_ == TRAIN && 0 == Caffe::worker_id())
    {
      caffe_gpu_add_scalar(this->blobs_[3]->count(), Dtype(1), 
                           this->blobs_[3]->mutable_gpu_diff());
    }
	}

	INSTANTIATE_LAYER_GPU_FUNCS(BatchReNormLayer);


}  // namespace caffe
