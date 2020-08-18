#include <algorithm>
#include <vector>

#include "caffe/layers/batch_renorm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void BatchReNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		BatchReNormParameter param = this->layer_param_.batch_renorm_param();
		moving_average_fraction_ = param.moving_average_fraction();
		use_global_stats_ = this->phase_ == TEST;
		if (param.has_use_global_stats())
			use_global_stats_ = param.use_global_stats();
		if (bottom[0]->num_axes() == 1)
			channels_ = 1;
		else
			channels_ = bottom[0]->shape(1);
		eps_ = param.eps();
		r_max_ = param.r_max();
		d_max_ = param.d_max();
		iter_size_ = param.iter_size();
		step_to_init_ = param.step_to_init();
		step_to_r_max_ = param.step_to_r_max();
		step_to_d_max_ = param.step_to_d_max();

		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			this->blobs_.resize(4);
			vector<int> sz;
			sz.push_back(channels_);
			this->blobs_[0].reset(new Blob<Dtype>(sz));
			this->blobs_[1].reset(new Blob<Dtype>(sz));
			sz[0] = 1;
			this->blobs_[2].reset(new Blob<Dtype>(sz));
			this->blobs_[3].reset(new Blob<Dtype>(sz));
			for (int i = 0; i < 4; ++i) {
				caffe_set(this->blobs_[i]->count(), Dtype(0),
					this->blobs_[i]->mutable_cpu_data());
			}
		}

		for (int i = 0; i < this->blobs_.size(); ++i) {
			if (this->layer_param_.param_size() == i) {
				ParamSpec* fixed_param_spec = this->layer_param_.add_param();
        fixed_param_spec->set_lr_mult(FIX_LEARNING_RATE);
				//fixed_param_spec->set_lr_mult(0.f);
				//fixed_param_spec->set_decay_mult(0.f);
        LOG(INFO) << "Init BRN param: " << i;
			}
			else {
        this->layer_param_.mutable_param(i)->set_lr_mult(FIX_LEARNING_RATE);
				//CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
				//	<< "Cannot configure batch normalization statistics as layer "
				//	<< "parameters.";
        LOG(INFO) << "Init BRN param: " << i;
			}
		}
	}

	template <typename Dtype>
	void BatchReNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (bottom[0]->num_axes() >= 1)
			CHECK_EQ(bottom[0]->shape(1), channels_);
		top[0]->ReshapeLike(*bottom[0]);

		vector<int> sz;
		sz.push_back(channels_);
		mean_.Reshape(sz);
		variance_.Reshape(sz);
    mean_glb_.Reshape(sz);
    variance_glb_.Reshape(sz);
    variance_back_.Reshape(sz);
		// temp.ReshapeLike(*bottom[0]);
		//x_norm.ReshapeLike(*bottom[0]);
    top[1]->ReshapeLike(*bottom[0]);    /* used as temp */
    top[2]->ReshapeLike(*bottom[0]);    /* used as x_norm */
    top[3]->ReshapeLike(*bottom[0]);    /* used as all_r */
    top[4]->ReshapeLike(*bottom[0]);    /* used as all_d */
		r_.Reshape(sz);
		d_.Reshape(sz);

		sz[0] = bottom[0]->shape(0);
		batch_sum_multiplier_.Reshape(sz);

		

		int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
		if (spatial_sum_multiplier_.num_axes() == 0 ||
			spatial_sum_multiplier_.shape(0) != spatial_dim) {
			sz[0] = spatial_dim;
			spatial_sum_multiplier_.Reshape(sz);
			Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
			caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
		}

		int numbychans = channels_*bottom[0]->shape(0);
		if (num_by_chans_.num_axes() == 0 ||
			num_by_chans_.shape(0) != numbychans) {
			sz[0] = numbychans;
			num_by_chans_.Reshape(sz);
			caffe_set(batch_sum_multiplier_.count(), Dtype(1),
				batch_sum_multiplier_.mutable_cpu_data());
		}
	}

	template <typename Dtype>
	void BatchReNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int num = bottom[0]->shape(0);
		int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);
		int iter = this->blobs_[3]->cpu_data()[0];
		int step = iter / iter_size_;
		bool first_iter_in_step = (iter%iter_size_ == 0);

    /* Use top[1] and top[2] for temp and x_norm */
    Blob<Dtype>& temp = *top[1];
    Blob<Dtype>& x_norm = *top[2];

		if (bottom[0] != top[0]) {
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
		}

		if (use_global_stats_) {
			// use the stored mean/variance estimates.
			const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
				0 : 1 / this->blobs_[2]->cpu_data()[0];
			caffe_cpu_scale(variance_.count(), scale_factor,
				this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
			caffe_cpu_scale(variance_.count(), scale_factor,
				this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
		}
		else {
			// compute mean
			caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), bottom_data,
				spatial_sum_multiplier_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data());
			caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
				mean_.mutable_cpu_data());
		}

		// subtract mean
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, -1, num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 1., top_data);

		if (!use_global_stats_) {
			// compute variance using var(X) = E((X-EX)^2)
			caffe_powx(top[0]->count(), top_data, Dtype(2),
				temp.mutable_cpu_data());  // (X-EX)^2
			caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), temp.cpu_data(),
				spatial_sum_multiplier_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data());
			caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
				variance_.mutable_cpu_data());  // E((X_EX)^2)
			
			if (step >= step_to_init_ && first_iter_in_step)
			{
				const Dtype scale_factor = 1. / this->blobs_[2]->cpu_data()[0];
				caffe_cpu_scale(variance_.count(), scale_factor, this->blobs_[0]->cpu_data(), this->blobs_[0]->mutable_cpu_diff());
				caffe_cpu_scale(variance_.count(), scale_factor, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff());
				caffe_add_scalar(variance_.count(), eps_, this->blobs_[1]->mutable_cpu_diff());
				caffe_powx(variance_.count(), this->blobs_[1]->cpu_diff(), Dtype(0.5), this->blobs_[1]->mutable_cpu_diff());
			}
			
			// compute and save moving average
			Dtype moving_average_fraction = first_iter_in_step ? moving_average_fraction_ : 1;
			this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction;
			this->blobs_[2]->mutable_cpu_data()[0] += 1;
			caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
				moving_average_fraction, this->blobs_[0]->mutable_cpu_data());
			int m = bottom[0]->count() / channels_;
			Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
			caffe_cpu_axpby(variance_.count(), bias_correction_factor,
				variance_.cpu_data(), moving_average_fraction,
				this->blobs_[1]->mutable_cpu_data());
		}

		// normalize variance
		caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
		caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
			variance_.mutable_cpu_data());
		
		// replicate variance to input size
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 0., temp.mutable_cpu_data());
		caffe_div(temp.count(), top_data, temp.cpu_data(), top_data);
		// TODO(cdoersch): The caching is only needed because later in-place layers
		//                 might clobber the data.  Can we skip this if they won't?
		caffe_copy(x_norm.count(), top_data,
			x_norm.mutable_cpu_data());

		if (!use_global_stats_ && step >= step_to_init_)
		{
			Dtype cur_r_max = __max(1, __min(1 + (step - step_to_init_ + 1)*(r_max_ - 1) / (step_to_r_max_ - step_to_init_), r_max_));
			Dtype cur_r_min = 1. / cur_r_max;
			Dtype cur_d_max = __max(0, __min((step - step_to_init_ + 1)*d_max_ / (step_to_d_max_ - step_to_init_), d_max_));
			Dtype cur_d_min = -cur_d_max;

			caffe_div(variance_.count(), variance_.cpu_data(), this->blobs_[1]->cpu_diff(), r_.mutable_cpu_data());
	
			caffe_copy(variance_.count(), mean_.cpu_data(), d_.mutable_cpu_data());
			caffe_cpu_axpby(variance_.count(), Dtype(-1), this->blobs_[0]->cpu_diff(), Dtype(1), d_.mutable_cpu_data());
			caffe_div(variance_.count(), d_.cpu_data(), this->blobs_[1]->cpu_diff(), d_.mutable_cpu_data());

			for (int i = 0; i < variance_.count(); ++i)
			{
				r_.mutable_cpu_data()[i] = __min(cur_r_max, __max(r_.mutable_cpu_data()[i], cur_r_min));
				d_.mutable_cpu_data()[i] = __min(cur_d_max, __max(d_.mutable_cpu_data()[i], cur_d_min));
			}

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
				batch_sum_multiplier_.cpu_data(), r_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data());
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
				spatial_dim, 1, 1., num_by_chans_.cpu_data(),
				spatial_sum_multiplier_.cpu_data(), 0., temp.mutable_cpu_diff());
			caffe_mul(temp.count(), top_data, temp.cpu_diff(), top_data);

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
				batch_sum_multiplier_.cpu_data(), d_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data());
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
				spatial_dim, 1, 1., num_by_chans_.cpu_data(),
				spatial_sum_multiplier_.cpu_data(), 0., x_norm.mutable_cpu_diff());
			caffe_add(temp.count(), top_data, x_norm.cpu_diff(), top_data);
		}
	}

	template <typename Dtype>
	void BatchReNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff;
		int iter = this->blobs_[3]->cpu_data()[0];
		int step = iter / iter_size_;

    /* Use top[1] and top[2] for temp and x_norm */
    Blob<Dtype>& temp = *top[1];
    Blob<Dtype>& x_norm = *top[2];

		if (bottom[0] != top[0]) {
			top_diff = top[0]->cpu_diff();
		}
		else {
			caffe_copy(x_norm.count(), top[0]->cpu_diff(), x_norm.mutable_cpu_diff());
			top_diff = x_norm.cpu_diff();
		}
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (use_global_stats_) {
			caffe_div(temp.count(), top_diff, temp.cpu_data(), bottom_diff);
			return;
		}
		const Dtype* top_data = x_norm.cpu_data();
		int num = bottom[0]->shape()[0];
		int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);
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
		caffe_mul(temp.count(), top_data, top_diff, bottom_diff);
		caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
			bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
			num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
			mean_.mutable_cpu_data());

		// reshape (broadcast) the above
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

		// sum(dE/dY \cdot Y) \cdot Y
		caffe_mul(temp.count(), top_data, bottom_diff, bottom_diff);

		// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
		caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
			top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
			num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
			mean_.mutable_cpu_data());
		// reshape (broadcast) the above to make
		// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
			spatial_dim, 1, 1., num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

		// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
		caffe_cpu_axpby(temp.count(), Dtype(1), top_diff,
			Dtype(-1. / (num * spatial_dim)), bottom_diff);

		// note: temp still contains sqrt(var(X)+eps), computed during the forward
		// pass.
		caffe_div(temp.count(), bottom_diff, temp.cpu_data(), bottom_diff);


		if (!use_global_stats_ && step >= step_to_init_)
		{
			caffe_mul(temp.count(), bottom_diff, temp.cpu_diff(), bottom_diff);
		}

		if (this->phase_ == TRAIN)
			this->blobs_[3]->mutable_cpu_data()[0] += 1;

	}


#ifdef CPU_ONLY
	STUB_GPU(BatchReNormLayer);
#endif

	INSTANTIATE_CLASS(BatchReNormLayer);
	REGISTER_LAYER_CLASS(BatchReNorm);
}  // namespace caffe
