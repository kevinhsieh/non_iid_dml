#ifndef CAFFE_BATCHRENORM_LAYER_HPP_
#define CAFFE_BATCHRENORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	
	template <typename Dtype>
	class BatchReNormLayer : public Layer<Dtype> {
	public:
		explicit BatchReNormLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BatchReNorm"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 5; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> mean_, variance_, temp_, x_norm_,r_,d_;
    Blob<Dtype> mean_glb_, variance_glb_, variance_back_;
		bool use_global_stats_;
		Dtype moving_average_fraction_;
		int channels_;
		Dtype eps_;

		Dtype r_max_, d_max_;
		int step_to_init_,step_to_r_max_, step_to_d_max_,iter_size_;		

		Blob<Dtype> batch_sum_multiplier_;
		Blob<Dtype> num_by_chans_;
		Blob<Dtype> spatial_sum_multiplier_;
	};

}  // namespace caffe

#endif  // CAFFE_BATCHRENORM_LAYER_HPP_
