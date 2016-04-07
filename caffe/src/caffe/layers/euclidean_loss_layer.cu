#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub<Dtype,Mtype>(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Mtype dot;
  caffe_gpu_dot<Dtype,Mtype>(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Mtype loss = dot / bottom[0]->num() / Mtype(2);
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss);
}

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Mtype sign((i == 0) ? 1. : -1.);
      const Mtype alpha = sign * Get<Mtype>(top[0]->cpu_diff()[0]) / bottom[i]->num();
      caffe_gpu_axpby<Dtype,Mtype>(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Mtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
