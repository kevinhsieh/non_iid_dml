#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(
      const SolverParameter& param, const PsConfig *ps_config = NULL)
      : SGDSolver<Dtype>(param, ps_config) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(
      const SolverParameter& param, const PsConfig *ps_config = NULL)
      : SGDSolver<Dtype>(param, ps_config) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(
      const SolverParameter& param, const PsConfig *ps_config = NULL)
      : SGDSolver<Dtype>(param, ps_config) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(
      const SolverParameter& param, const PsConfig *ps_config = NULL)
      : SGDSolver<Dtype>(param, ps_config) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(
      const SolverParameter& param, const PsConfig *ps_config = NULL)
      : SGDSolver<Dtype>(param, ps_config) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
