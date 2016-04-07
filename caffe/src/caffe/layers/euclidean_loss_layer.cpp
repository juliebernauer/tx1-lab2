#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Reshape(
  const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  LossLayer<Dtype,Mtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub<Dtype,Mtype>(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Mtype dot = caffe_cpu_dot<Dtype,Mtype>(count, diff_.cpu_data(), diff_.cpu_data());
  Mtype loss = dot / bottom[0]->num() / Mtype(2.f);
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss);
}

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Mtype sign(i == 0 ? 1.f : -1.f);
      const Mtype alpha(sign * Get<Mtype>(top[0]->cpu_diff()[0] / bottom[i]->num()));
      caffe_cpu_axpby<Dtype,Mtype>(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Mtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
