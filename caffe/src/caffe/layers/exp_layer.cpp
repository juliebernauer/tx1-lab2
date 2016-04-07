#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ExpLayer<Dtype,Mtype>::LayerSetUp(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
  NeuronLayer<Dtype,Mtype>::LayerSetUp(bottom, top);
  const Mtype base(this->layer_param_.exp_param().base());
  if (base != Mtype(-1)) {
    CHECK_GT(base, 0) << "base must be strictly positive.";
  }
  // If base == -1, interpret the base as e and set log_base = 1 exactly.
  // Otherwise, calculate its log explicitly.
  const Mtype log_base(base == -1. ? 1. : log(base));
  CHECK(!isnan(log_base))
      << "NaN result: log(base) = log(" << base << ") = " << log_base;
  CHECK(!isinf(log_base))
      << "Inf result: log(base) = log(" << base << ") = " << log_base;
  const Mtype input_scale(this->layer_param_.exp_param().scale());
  const Mtype input_shift(this->layer_param_.exp_param().shift());
  inner_scale_ = log_base * input_scale;
  outer_scale_ = input_shift == 0. ? 1. : pow(base, input_shift);
}

template <typename Dtype, typename Mtype>
void ExpLayer<Dtype,Mtype>::Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (inner_scale_ == Mtype(1)) {
    caffe_exp<Dtype,Mtype>(count, bottom_data, top_data);
  } else {
    caffe_cpu_scale<Dtype,Mtype>(count, inner_scale_, bottom_data, top_data);
    caffe_exp<Dtype,Mtype>(count, top_data, top_data);
  }
  if (outer_scale_ != Mtype(1)) {
    caffe_scal<Dtype,Mtype>(count, outer_scale_, top_data);
  }
}

template <typename Dtype, typename Mtype>
void ExpLayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_mul<Dtype,Mtype>(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != Mtype(1)) {
    caffe_scal<Dtype,Mtype>(count, inner_scale_, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ExpLayer);
#endif

INSTANTIATE_CLASS(ExpLayer);
REGISTER_LAYER_CLASS(Exp);

}  // namespace caffe
